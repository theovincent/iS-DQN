from typing import Tuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


class iSHLDQN:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        n_bellman_iterations: int,
        n_bins: int,
        features: list,
        layer_norm: bool,
        architecture_type: str,
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        data_to_update: int,
        target_update_frequency: int,
        target_sync_frequency: int,
        min_value: float,
        max_value: float,
        sigma: float,
        adam_eps: float = 1e-8,
    ):
        self.n_bellman_iterations = n_bellman_iterations
        self.n_actions = n_actions
        self.n_bins = n_bins
        self.last_idx_mlp = len(features) if architecture_type == "fc" else len(features) - 3
        self.network = DQNNet(
            features, architecture_type, 2 * self.n_bellman_iterations * n_actions * self.n_bins, layer_norm
        )
        # 2 * self.n_bellman_iterations = [\bar{Q_0}, ..., \bar{Q_K-1}, Q_1, ..., Q_K]
        self.network.apply_fn = lambda params, state: jnp.squeeze(
            self.network.apply(params, state).reshape((-1, 2 * self.n_bellman_iterations, n_actions, self.n_bins))
        )
        self.params = self.network.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.data_to_update = data_to_update
        self.target_update_frequency = target_update_frequency
        self.target_sync_frequency = target_sync_frequency
        self.cumulated_losses = np.zeros(self.n_bellman_iterations)
        self.cumulated_unsupported_probs = np.zeros(self.n_bellman_iterations)
        self.support = jnp.linspace(min_value, max_value, self.n_bins + 1, dtype=jnp.float32)
        self.bin_centers = (self.support[:-1] + self.support[1:]) / 2
        self.clip_target = lambda target: jnp.clip(target, min_value, max_value)
        self.sigma = sigma

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.data_to_update == 0:
            batch_samples = replay_buffer.sample()

            self.params, self.optimizer_state, losses, unsupported_probs = self.learn_on_batch(
                self.params, self.optimizer_state, batch_samples
            )
            self.cumulated_losses += losses
            self.cumulated_unsupported_probs += unsupported_probs

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            # Window shift
            self.params = self.shift_params(self.params)

            logs = {
                "loss": self.cumulated_losses.mean() / (self.target_update_frequency / self.data_to_update),
            }
            for idx_network in range(min(self.n_bellman_iterations, 5)):
                logs[f"networks/{idx_network}_loss"] = self.cumulated_losses[idx_network] / (
                    self.target_update_frequency / self.data_to_update
                )
                logs[f"networks/{idx_network}_unsupported_prob"] = self.cumulated_unsupported_probs[idx_network] / (
                    self.target_update_frequency / self.data_to_update
                )
            self.cumulated_losses = np.zeros_like(self.cumulated_losses)
            self.cumulated_unsupported_probs = np.zeros_like(self.cumulated_unsupported_probs)

            return True, logs

        if step % self.target_sync_frequency == 0:
            self.target_params = self.sync_target_params(self.params)

        return False, {}

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(self, params: FrozenDict, optimizer_state, batch_samples):
        grad_loss, (losses, unsupported_probs) = jax.grad(self.loss_on_batch, has_aux=True)(params, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, losses, unsupported_probs

    def loss_on_batch(self, params: FrozenDict, samples):
        batch_size = samples.state.shape[0]
        # shape (2 * batch_size, 2 * n_bellman_iterations, n_actions, n_bins)
        all_q_logits = self.network.apply_fn(params, jnp.concatenate((samples.state, samples.next_state)))
        # shape (batch_size, n_bellman_iterations, n_bins)
        q_logits = jax.vmap(lambda q_value, action: q_value[:, action])(
            all_q_logits[:batch_size, self.n_bellman_iterations :], samples.action
        )
        # shape (batch_size, n_bellman_iterations)
        targets = jax.vmap(self.compute_target)(samples, all_q_logits[batch_size:, : -self.n_bellman_iterations])
        # shape (batch_size, n_bellman_iterations, n_bins) | (batch_size, n_bellman_iterations)
        projected_targets, unsupported_probs = jax.vmap(jax.vmap(self.project_target_on_support))(targets)
        stop_grad_projected_targets = jax.lax.stop_gradient(projected_targets)

        # shape (batch_size, n_bellman_iterations)
        cross_entropies = optax.softmax_cross_entropy(q_logits, stop_grad_projected_targets)
        return cross_entropies.mean(axis=0).sum(), (cross_entropies.mean(axis=0), unsupported_probs.mean(axis=0))

    def compute_target(self, sample: ReplayElement, next_q_logits: jax.Array):
        # computes the target value for single sample
        # We first compute the probabilities by applying the softmax on the last axis (bin axis).
        # Then, we compute the expectation by multiplying with the bin centers.
        next_values = jax.nn.softmax(next_q_logits, axis=-1) @ self.bin_centers
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(
            next_values, axis=-1
        )

    def project_target_on_support(self, target: jax.Array) -> Tuple[jax.Array, jax.Array]:
        # We use the error function. It is linked with the cumulative distribution function of a gaussian distribution.
        erf_support = jax.scipy.special.erf((self.support - self.clip_target(target)) / (jnp.sqrt(2) * self.sigma))
        # We also output the probability mass which does not lie on the support
        # CDF(min support) + 1 - CDF(max support)
        return (
            (erf_support[1:] - erf_support[:-1]) / (erf_support[-1] - erf_support[0] + 1e-9),
            erf_support[0] / 2 + 1 - erf_support[-1] / 2,
        )

    @partial(jax.jit, static_argnames="self")
    def shift_params(self, params):
        # Shift the last weight matrix with shape (last_feature, 2 x n_bellman_iterations x n_actions)
        # Reminder: 2 * self.n_bellman_iterations = [\bar{Q_0}, ..., \bar{Q_K-1}, Q_1, ..., Q_K]
        # Here we shifting: \bar{Q_i} <- Q_i+1
        kernel = params["params"][f"Dense_{self.last_idx_mlp}"]["kernel"]
        params["params"][f"Dense_{self.last_idx_mlp}"]["kernel"] = kernel.at[
            :, : self.n_bellman_iterations * self.n_actions * self.n_bins
        ].set(kernel[:, self.n_bellman_iterations * self.n_actions * self.n_bins :])

        # Shift the last bias vector with shape (2 x n_bellman_iterations x n_actions)
        bias = params["params"][f"Dense_{self.last_idx_mlp}"]["bias"]
        params["params"][f"Dense_{self.last_idx_mlp}"]["bias"] = bias.at[
            : self.n_bellman_iterations * self.n_actions * self.n_bins
        ].set(bias[self.n_bellman_iterations * self.n_actions * self.n_bins :])

        return params

    @partial(jax.jit, static_argnames="self")
    def sync_target_params(self, params):
        # Synchronize the last weight matrix with shape (last_feature, 2 x n_bellman_iterations x n_actions)
        # Reminder: 2 * self.n_bellman_iterations = [\bar{Q_0}, ..., \bar{Q_K-1}, Q_1, ..., Q_K]
        # Here we synchronize: \bar{Q_i} <- Q_i
        kernel = params["params"][f"Dense_{self.last_idx_mlp}"]["kernel"]
        params["params"][f"Dense_{self.last_idx_mlp}"]["kernel"] = kernel.at[
            :, self.n_actions * self.n_bins : self.n_bellman_iterations * self.n_actions * self.n_bins
        ].set(kernel[:, self.n_bellman_iterations * self.n_actions * self.n_bins : -self.n_actions * self.n_bins])

        # Synchronize the last bias vector with shape (2 x n_bellman_iterations x n_actions)
        bias = params["params"][f"Dense_{self.last_idx_mlp}"]["bias"]
        params["params"][f"Dense_{self.last_idx_mlp}"]["bias"] = bias.at[
            self.n_actions * self.n_bins : self.n_bellman_iterations * self.n_actions * self.n_bins
        ].set(bias[self.n_bellman_iterations * self.n_actions * self.n_bins : -self.n_actions * self.n_bins])

        return params

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.Array):
        idx_network = jax.random.randint(key, (), 0, self.n_bellman_iterations)
        # shape (2 * n_bellman_iterations, n_actions, n_bins)
        q_logits = self.network.apply_fn(params, state)

        # computes the best action for a single state
        # We first compute the probabilities by applying the softmax on the last axis (bin axis).
        # Then, we compute the expectation by multiplying with the bin centers.
        return jnp.argmax(jax.nn.softmax(q_logits[self.n_bellman_iterations + idx_network]) @ self.bin_centers)

    def get_model(self):
        return {"params": self.params}
