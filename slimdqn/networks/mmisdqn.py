from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


class MMiSDQN:
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
        omega: float,
        adam_eps: float = 1e-8,
    ):
        self.n_bellman_iterations = n_bellman_iterations
        self.n_actions = n_actions
        self.last_idx_mlp = len(features) if architecture_type == "fc" else len(features) - 3
        self.network = DQNNet(features, architecture_type, (1 + self.n_bellman_iterations) * n_actions, layer_norm)

        # 1 + self.n_bellman_iterations = [\bar{Q_0}, Q_1, ..., Q_K]
        def apply(params, state):
            q_values, batch_stats = self.network.apply(params, state, mutable=["batch_stats"])
            return q_values.reshape((-1, 1 + self.n_bellman_iterations, n_actions)), batch_stats

        self.network.apply_fn = apply
        self.params = self.network.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.data_to_update = data_to_update
        self.target_update_frequency = target_update_frequency
        self.omega = omega
        self.cumulated_losses = np.zeros(self.n_bellman_iterations)

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.data_to_update == 0:
            batch_samples = replay_buffer.sample()

            self.params, self.optimizer_state, losses = self.learn_on_batch(
                self.params, self.optimizer_state, batch_samples
            )
            self.cumulated_losses += losses

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            # Window shift
            self.params = self.shift_params(self.params)

            logs = {
                "loss": np.mean(self.cumulated_losses) / (self.target_update_frequency / self.data_to_update),
            }
            for idx_network in range(min(self.n_bellman_iterations, 5)):
                logs[f"networks/{idx_network}_loss"] = self.cumulated_losses[idx_network] / (
                    self.target_update_frequency / self.data_to_update
                )
            self.cumulated_losses = np.zeros_like(self.cumulated_losses)

            return True, logs

        return False, {}

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(self, params: FrozenDict, optimizer_state, batch_samples):
        grad_loss, (losses, batch_stats) = jax.grad(self.loss_on_batch, has_aux=True)(params, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)
        if self.network.batch_norm:
            params["batch_stats"] = batch_stats["batch_stats"]

        return params, optimizer_state, losses

    def loss_on_batch(self, params: FrozenDict, samples):
        batch_size = samples.state.shape[0]
        # shape (2 * batch_size, 1 + n_bellman_iterations, n_actions) | Dict
        all_q_values, batch_stats = self.network.apply_fn(params, jnp.concatenate((samples.state, samples.next_state)))
        # shape (batch_size, n_bellman_iterations)
        q_values = jax.vmap(lambda q_value, action: q_value[:, action])(all_q_values[:batch_size, 1:], samples.action)
        targets = jax.vmap(self.compute_target)(samples, all_q_values[batch_size:, :-1])
        stop_grad_targets = jax.lax.stop_gradient(targets)

        # shape (batch_size, n_bellman_iterations)
        td_losses = jnp.square(q_values - stop_grad_targets)
        return td_losses.mean(axis=0).sum(), (td_losses.mean(axis=0), batch_stats)

    def compute_target(self, sample: ReplayElement, next_q_values: jax.Array):
        # shape of next_q_values (n_bellman_iterations, next_states, n_actions)
        mm_q = jnp.log(jnp.mean(jnp.exp(self.omega * next_q_values), axis=-1) + 1e-9) / self.omega
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * mm_q

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
        idx_network = jax.random.randint(key, (), 0, self.n_bellman_iterations)
        q_values = self.network.apply(params, state, use_running_average=True).reshape(
            (1 + self.n_bellman_iterations, self.n_actions)
        )

        # computes the best action for a single state from a uniformly chosen online network
        return jnp.argmax(q_values[1 + idx_network])

    def get_model(self):
        return {"params": self.params}
