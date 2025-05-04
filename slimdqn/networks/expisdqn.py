from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


class ExpiSDQN:
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
        target_sync_frequency: int,
        adam_eps: float = 1e-8,
    ):
        self.n_bellman_iterations = n_bellman_iterations
        self.n_actions = n_actions
        self.last_idx_mlp = len(features) if architecture_type == "fc" else len(features) - 3
        self.network = DQNNet(features, architecture_type, 2 * self.n_bellman_iterations * n_actions, layer_norm)
        # 2 * self.n_bellman_iterations = [\bar{Q_0}, ..., \bar{Q_K-1}, Q_1, ..., Q_K]
        self.network.apply_fn = lambda params, state: jnp.squeeze(
            self.network.apply(params, state).reshape((-1, 2 * self.n_bellman_iterations, n_actions))
        )
        self.params = self.network.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)
        self.target_params = self.params

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.data_to_update = data_to_update
        self.target_update_frequency = target_update_frequency
        self.target_sync_frequency = target_sync_frequency
        self.cumulated_losses = np.zeros(self.n_bellman_iterations)
        self.cumulated_targets_change = np.zeros(self.n_bellman_iterations)

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.data_to_update == 0:
            batch_samples = replay_buffer.sample()

            self.params, self.optimizer_state, losses, targets_change = self.learn_on_batch(
                self.params, self.target_params, self.optimizer_state, batch_samples
            )
            self.cumulated_losses += losses
            self.cumulated_targets_change += targets_change

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            # Window shift
            self.target_params = self.params.copy()
            self.params = self.shift_params(self.params)

            logs = {
                "loss": np.mean(self.cumulated_losses) / (self.target_update_frequency / self.data_to_update),
            }
            for idx_network in range(min(self.n_bellman_iterations, 5)):
                logs[f"networks/{idx_network}_loss"] = self.cumulated_losses[idx_network] / (
                    self.target_update_frequency / self.data_to_update
                )
                logs[f"networks/{idx_network}_target_change"] = self.cumulated_targets_change / (
                    self.target_update_frequency / self.data_to_update
                )
            self.cumulated_losses = np.zeros_like(self.cumulated_losses)
            self.cumulated_targets_change = np.zeros_like(self.cumulated_targets_change)

            return True, logs

        if step % self.target_sync_frequency == 0:
            self.target_params = self.sync_target_params(self.params)

        return False, {}

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(self, params: FrozenDict, params_target: FrozenDict, optimizer_state, batch_samples):
        grad_loss, (losses, targets_change) = jax.grad(self.loss_on_batch, has_aux=True)(
            params, params_target, batch_samples
        )
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, losses, targets_change

    def loss_on_batch(self, params: FrozenDict, params_target: FrozenDict, samples):
        batch_size = samples.state.shape[0]
        # shape (2 * batch_size, 2 * n_bellman_iterations, n_actions)
        all_q_values = self.network.apply_fn(params, jnp.concatenate((samples.state, samples.next_state)))
        # shape (batch_size, n_bellman_iterations)
        q_values = jax.vmap(lambda q_value, action: q_value[:, action])(
            all_q_values[:batch_size, self.n_bellman_iterations :], samples.action
        )
        targets = jax.vmap(self.compute_target)(samples, all_q_values[batch_size:, : self.n_bellman_iterations])
        stop_grad_targets = jax.lax.stop_gradient(targets)

        frozen_targets = jax.vmap(self.compute_target)(
            samples, self.network.apply_fn(params_target, samples.next_state)[:, : self.n_bellman_iterations]
        )
        targets_change = (targets - frozen_targets) / (frozen_targets + 1e-9)

        # shape (batch_size, n_bellman_iterations)
        td_losses = jnp.square(q_values - stop_grad_targets)
        return td_losses.mean(axis=0).sum(), td_losses.mean(axis=0), targets_change.mean(axis=0)

    def compute_target(self, sample: ReplayElement, next_q_values: jax.Array):
        # shape of next_q_values (n_bellman_iterations, next_states, n_actions)
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(
            next_q_values, axis=-1
        )

    @partial(jax.jit, static_argnames="self")
    def shift_params(self, params):
        # Shift the last weight matrix with shape (last_feature, 2 x n_bellman_iterations x n_actions)
        # Reminder: 2 * self.n_bellman_iterations = [\bar{Q_0}, ..., \bar{Q_K-1}, Q_1, ..., Q_K]
        # Here we shifting: \bar{Q_i} <- \bar{Q_i+1} and Q_i <- Q_i+1
        kernel = params["params"][f"Dense_{self.last_idx_mlp}"]["kernel"]
        kernel = kernel.at[:, : (self.n_bellman_iterations - 1) * self.n_actions].set(
            kernel[:, self.n_actions : self.n_bellman_iterations * self.n_actions]
        )
        params["params"][f"Dense_{self.last_idx_mlp}"]["kernel"] = kernel.at[
            :, self.n_bellman_iterations * self.n_actions : -self.n_actions
        ].set(kernel[:, (self.n_bellman_iterations + 1) * self.n_actions :])

        # Shift the last bias vector with shape (2 x n_bellman_iterations x n_actions)
        bias = params["params"][f"Dense_{self.last_idx_mlp}"]["bias"]
        bias = bias.at[: (self.n_bellman_iterations - 1) * self.n_actions].set(
            bias[self.n_actions : self.n_bellman_iterations * self.n_actions]
        )
        params["params"][f"Dense_{self.last_idx_mlp}"]["bias"] = bias.at[
            self.n_bellman_iterations * self.n_actions : -self.n_actions
        ].set(bias[(self.n_bellman_iterations + 1) * self.n_actions :])

        return params

    @partial(jax.jit, static_argnames="self")
    def sync_target_params(self, params):
        # Synchronize the last weight matrix with shape (last_feature, 2 x n_bellman_iterations x n_actions)
        # Reminder: 2 * self.n_bellman_iterations = [\bar{Q_0}, ..., \bar{Q_K-1}, Q_1, ..., Q_K]
        # Here we synchronize: \bar{Q_i} <- Q_i
        kernel = params["params"][f"Dense_{self.last_idx_mlp}"]["kernel"]
        params["params"][f"Dense_{self.last_idx_mlp}"]["kernel"] = kernel.at[
            :, self.n_actions : self.n_bellman_iterations * self.n_actions
        ].set(kernel[:, self.n_bellman_iterations * self.n_actions : -self.n_actions])

        # Synchronize the last bias vector with shape (2 x n_bellman_iterations x n_actions)
        bias = params["params"][f"Dense_{self.last_idx_mlp}"]["bias"]
        params["params"][f"Dense_{self.last_idx_mlp}"]["bias"] = bias.at[
            self.n_actions : self.n_bellman_iterations * self.n_actions
        ].set(bias[self.n_bellman_iterations * self.n_actions : -self.n_actions])

        return params

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.Array):
        idx_network = jax.random.randint(key, (), 0, self.n_bellman_iterations)

        # computes the best action for a single state from a uniformly chosen online network
        return jnp.argmax(self.network.apply_fn(params, state)[self.n_bellman_iterations + idx_network])

    def get_model(self):
        return {"params": self.params}
