from typing import Tuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


@jax.jit
def sync_target_params(params, target_params):
    # Each target network is synchronized to the online network it represents
    # \bar{\theta}_k <- \theta_k, i.e., target_params[k] <- params[k-1]
    return jax.tree.map(lambda param, target_param: target_param.at[1:].set(param[:-1]), params, target_params)


@jax.jit
def shift_params(params):
    # Each online network is updated to the following online network
    # \theta_k <- \theta_{k + 1}, i.e., params[k] <- params[k + 1]
    return jax.tree.map(lambda param: param.at[:-1].set(param[1:]), params)


class iHLDQN:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        n_networks: int,
        n_bins: int,
        features: list,
        architecture_type: str,
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_frequency: int,
        target_sync_frequency: int,
        min_value: float,
        max_value: float,
        sigma: float,
        adam_eps: float = 1e-8,
    ):
        self.n_networks = n_networks
        self.n_bins = n_bins
        self.network = DQNNet(features, architecture_type, n_actions * self.n_bins)
        self.network.apply_fn = lambda params, state: self.network.apply(params, state).reshape(
            (n_actions, self.n_bins)
        )
        # Create K online parameters
        # params = [\theta_1, \theta_2, ..., \theta_K]
        self.params = jax.vmap(self.network.init, in_axes=(0, None))(
            jax.random.split(key, self.n_networks), jnp.zeros(observation_dim, dtype=jnp.float32)
        )

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)
        # Create K target parameters
        # target_params = [\bar{\theta}_0, \bar{\theta}_1, ..., \bar{\theta}_{K-1}]
        self.target_params = self.params

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_frequency = target_update_frequency
        self.target_sync_frequency = target_sync_frequency
        self.cumulated_losses = np.zeros(self.n_networks)
        self.cumulated_unsupported_probs = np.zeros(self.n_networks)
        self.support = jnp.linspace(min_value, max_value, self.n_bins + 1, dtype=jnp.float32)
        self.bin_centers = (self.support[:-1] + self.support[1:]) / 2
        self.sigma = sigma

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.update_to_data == 0:
            batch_samples = replay_buffer.sample()

            self.params, self.optimizer_state, losses, unsupported_probs = self.learn_on_batch(
                self.params, self.target_params, self.optimizer_state, batch_samples
            )
            self.cumulated_losses += losses
            self.cumulated_unsupported_probs += unsupported_probs

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            # Each target network is updated to its respective online network
            # \bar{\theta}_k <- \theta_{k + 1}, i.e., target_params[k] <- params[k]
            self.target_params = self.params.copy()
            # Window shift
            self.params = shift_params(self.params)

            logs = {"loss": self.cumulated_losses.mean() / (self.target_update_frequency / self.update_to_data)}
            for idx_network in range(self.n_networks):
                logs[f"networks/{idx_network}_loss"] = self.cumulated_losses[idx_network] / (
                    self.target_update_frequency / self.update_to_data
                )
                logs[f"networks/{idx_network}_unsupported_prob"] = self.cumulated_unsupported_probs[idx_network] / (
                    self.target_update_frequency / self.update_to_data
                )
            self.cumulated_losses = np.zeros_like(self.cumulated_losses)
            self.cumulated_unsupported_probs = np.zeros_like(self.cumulated_unsupported_probs)

            return True, logs

        if step % self.target_sync_frequency == 0:
            self.target_params = sync_target_params(self.params, self.target_params)

        return False, {}

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self,
        params: FrozenDict,
        params_target: FrozenDict,
        optimizer_state,
        batch_samples,
    ):
        grad_loss, (losses, unsupported_probs) = jax.grad(self.loss_on_batch, has_aux=True)(
            params, params_target, batch_samples
        )
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, losses, unsupported_probs

    def loss_on_batch(self, params: FrozenDict, params_target: FrozenDict, samples):
        # map over params, then map over samples
        losses, unsupported_probs = jax.vmap(jax.vmap(self.loss, in_axes=(None, None, 0)), in_axes=(0, 0, None))(
            params, params_target, samples
        )
        return losses.mean(), (losses.mean(axis=1), unsupported_probs.mean(axis=1))

    def loss(self, params: FrozenDict, params_target: FrozenDict, sample: ReplayElement) -> Tuple[jax.Array, jax.Array]:
        # computes the loss for a single sample
        target = self.compute_target(params_target, sample)
        q_logits = self.network.apply_fn(params, sample.state)[sample.action]
        projected_target, unsupported_prob = self.project_target_on_support(target)
        return optax.softmax_cross_entropy(q_logits, projected_target), unsupported_prob

    def compute_target(self, params: FrozenDict, sample: ReplayElement):
        # computes the target value for single sample
        # We first compute the probabilities by applying the softmax on the last axis (bin axis).
        # Then, we compute the expectation by multiplying with the bin centers.
        next_values = jax.nn.softmax(self.network.apply_fn(params, sample.next_state)) @ self.bin_centers
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(next_values)

    def project_target_on_support(self, target: jax.Array) -> jax.Array:
        # We use the error function. It is linked with the cumulative distribution function of a gaussian distribution.
        erf_support = jax.scipy.special.erf((self.support - target) / (jnp.sqrt(2) * self.sigma))
        # We also output the probability mass which does not lie on the support
        # CDF(min support) + 1 - CDF(max support)
        return (
            (erf_support[1:] - erf_support[:-1]) / (erf_support[-1] - erf_support[0] + 1e-6),
            erf_support[0] / 2 + 1 - erf_support[-1] / 2,
        )

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.Array):
        idx_params = jax.random.randint(key, (), 0, self.n_networks)
        selected_params = jax.tree.map(lambda param: param[idx_params], params)

        # computes the best action for a single state
        # We first compute the probabilities by applying the softmax on the last axis (bin axis).
        # Then, we compute the expectation by multiplying with the bin centers.
        return jnp.argmax(jax.nn.softmax(self.network.apply_fn(selected_params, state)) @ self.bin_centers)

    def get_model(self):
        return {"params": self.params}
