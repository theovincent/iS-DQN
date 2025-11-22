from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


class MetaiSDQN:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        n_bellman_iterations: int,
        features: list,
        layer_norm: bool,
        architecture_type: str,
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        data_to_update: int,
        target_update_frequency: int,
        adam_eps: float = 1e-8,
    ):
        self.n_bellman_iterations = n_bellman_iterations
        self.n_actions = n_actions
        self.last_idx_mlp = len(features) if architecture_type == "fc" else len(features) - 3
        self.network = DQNNet(features, architecture_type, (1 + self.n_bellman_iterations) * n_actions, layer_norm)

        # 1 + self.n_bellman_iterations = [\bar{Q_0}, Q_1, ..., Q_K]
        def apply(params, state):
            return self.network.apply(params, state).reshape((-1, 1 + self.n_bellman_iterations, n_actions))

        self.meta_params = {"alpha_logits": jnp.zeros(self.n_bellman_iterations, dtype=jnp.float32)}
        self.network.apply_fn = apply
        self.params = self.network.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.meta_optimizer = optax.adam(learning_rate)
        self.meta_optimizer_state = self.meta_optimizer.init(self.meta_params)
        # eps_root=1e-9 so that the gradient over adam does not output nans
        self.optimizer = optax.adam(learning_rate, eps=adam_eps, eps_root=1e-9)
        self.optimizer_state = self.optimizer.init(self.params)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.data_to_update = data_to_update
        self.target_update_frequency = target_update_frequency
        self.cumulated_losses = np.zeros(self.n_bellman_iterations)
        self.cumulated_alphas = np.zeros(self.n_bellman_iterations)

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.data_to_update == 0:
            batch_samples = replay_buffer.sample()

            self.meta_params, self.params, self.meta_optimizer_state, self.optimizer_state, losses = (
                self.meta_learn_on_batch(
                    self.meta_params, self.params, self.meta_optimizer_state, self.optimizer_state, batch_samples
                )
            )
            self.cumulated_losses += losses
            self.cumulated_alphas += jax.nn.softmax(self.meta_params["alpha_logits"]) * self.n_bellman_iterations

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            # Window shift
            self.params = self.shift_params(self.params)

            logs = {
                "loss": np.mean(self.cumulated_losses) / (self.target_update_frequency / self.data_to_update),
            }
            for idx_network in range(self.n_bellman_iterations):
                logs[f"networks/{idx_network}_loss"] = self.cumulated_losses[idx_network] / (
                    self.target_update_frequency / self.data_to_update
                )
                logs[f"networks/{idx_network}_alphas"] = self.cumulated_alphas[idx_network] / (
                    self.target_update_frequency / self.data_to_update
                )
            self.cumulated_losses = np.zeros_like(self.cumulated_losses)
            self.cumulated_alphas = np.zeros_like(self.cumulated_losses)

            return True, logs

        return False, {}

    @partial(jax.jit, static_argnames="self")
    def meta_learn_on_batch(
        self, meta_params: FrozenDict, params: FrozenDict, meta_optimizer_state, optimizer_state, batch_samples
    ):
        meta_grad_loss, (params, optimizer_state, td_losses) = jax.grad(self.meta_loss_on_batch, has_aux=True)(
            meta_params, params, optimizer_state, batch_samples
        )
        meta_updates, meta_optimizer_state = self.meta_optimizer.update(meta_grad_loss, meta_optimizer_state)
        meta_params = optax.apply_updates(meta_params, meta_updates)

        return meta_params, params, meta_optimizer_state, optimizer_state, td_losses

    def meta_loss_on_batch(self, meta_params: FrozenDict, params: FrozenDict, optimizer_state, samples):
        grad_loss, _ = jax.grad(self.loss_on_batch, has_aux=True)(params, meta_params, samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        meta_loss, td_losses = self.loss_on_batch(
            params,
            {"alpha_logits": jnp.ones(self.n_bellman_iterations, dtype=jnp.float32)},
            samples,
        )

        return meta_loss, (params, optimizer_state, td_losses)

    def loss_on_batch(self, params: FrozenDict, meta_params: FrozenDict, samples):
        batch_size = samples.state.shape[0]
        # shape (2 * batch_size, 1 + n_bellman_iterations, n_actions) | Dict
        all_q_values = self.network.apply_fn(params, jnp.concatenate((samples.state, samples.next_state)))
        # shape (batch_size, n_bellman_iterations)
        q_values = jax.vmap(lambda q_value, action: q_value[:, action])(all_q_values[:batch_size, 1:], samples.action)
        targets = jax.vmap(self.compute_target)(samples, all_q_values[batch_size:, :-1])
        stop_grad_targets = jax.lax.stop_gradient(targets)

        # shape (batch_size, n_bellman_iterations)
        td_losses = jnp.square(q_values - stop_grad_targets)
        alphas = jax.nn.softmax(meta_params["alpha_logits"]) * self.n_bellman_iterations
        return alphas @ td_losses.mean(axis=0), td_losses.mean(axis=0)

    def compute_target(self, sample: ReplayElement, next_q_values: jax.Array):
        # shape of next_q_values (n_bellman_iterations, next_states, n_actions)
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(
            next_q_values, axis=-1
        )

    @partial(jax.jit, static_argnames="self")
    def shift_params(self, params):
        # Shift the last weight matrix with shape (last_feature, (1 + n_bellman_iterations) x n_actions)
        # Reminder: 1 + self.n_bellman_iterations = [\bar{Q_0}, Q_1, ..., Q_K]
        # Here we shifting: \bar{Q_i} <- Q_i+1
        kernel = params["params"][f"Dense_{self.last_idx_mlp}"]["kernel"]
        params["params"][f"Dense_{self.last_idx_mlp}"]["kernel"] = kernel.at[:, : -self.n_actions].set(
            kernel[:, self.n_actions :]
        )

        # Shift the last bias vector with shape ((1 + n_bellman_iterations) x n_actions)
        bias = params["params"][f"Dense_{self.last_idx_mlp}"]["bias"]
        params["params"][f"Dense_{self.last_idx_mlp}"]["bias"] = bias.at[: -self.n_actions].set(bias[self.n_actions :])

        return params

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.Array):
        q_values = self.network.apply_fn(params["params"], state)[0][1:]
        idx_network = jax.random.choice(
            key, jnp.arange(0, self.n_bellman_iterations), (), p=jax.nn.softmax(params["meta_params"]["alpha_logits"])
        )
        return jnp.argmax(q_values[idx_network])

    @property
    def all_params(self):
        return {"params": self.params, "meta_params": self.meta_params}

    def get_model(self):
        return {"params": self.params}
