from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


class TFDQN:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        features: list,
        layer_norm: bool,
        batch_norm: bool,
        architecture_type: str,
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        data_to_update: int,
        target_update_frequency: int,
        adam_eps: float = 1e-8,
    ):
        self.network = DQNNet(features, architecture_type, n_actions, layer_norm, batch_norm)
        self.params = self.network.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.data_to_update = data_to_update
        self.target_update_frequency = target_update_frequency
        self.cumulated_loss = 0

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.data_to_update == 0:
            batch_samples = replay_buffer.sample()

            self.params, self.optimizer_state, loss = self.learn_on_batch(
                self.params, self.optimizer_state, batch_samples
            )
            self.cumulated_loss += loss

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            logs = {"loss": self.cumulated_loss / (self.target_update_frequency / self.data_to_update)}
            self.cumulated_loss = 0

            return True, logs
        return False, {}

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(self, params: FrozenDict, optimizer_state, batch_samples):
        (loss, batch_stats), grad_loss = jax.value_and_grad(self.loss_on_batch, has_aux=True)(params, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)
        if self.network.batch_norm:
            params["batch_stats"] = batch_stats["batch_stats"]

        return params, optimizer_state, loss

    def loss_on_batch(self, params: FrozenDict, samples):
        batch_size = samples.state.shape[0]
        # shape (2 * batch_size, n_actions)
        all_q_values, batch_stats = self.network.apply(
            params, jnp.concatenate((samples.state, samples.next_state)), mutable=["batch_stats"]
        )
        q_values = jax.vmap(lambda q_value, action: q_value[action])(all_q_values[:batch_size], samples.action)
        targets = self.compute_target(samples, all_q_values[batch_size:])
        stop_grad_targets = jax.lax.stop_gradient(targets)
        # shape (batch_size, n_bellman_iterations)
        td_losses = jnp.square(q_values - stop_grad_targets)

        return td_losses.mean(), batch_stats

    def compute_target(self, sample: ReplayElement, next_q_values: jax.Array):
        # shape of next_q_values (next_states, n_actions)
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(
            next_q_values, axis=-1
        )

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, **kwargs):
        # computes the best action for a single state
        return jnp.argmax(self.network.apply(params, state, use_running_average=True))

    def get_model(self):
        return {"params": self.params}
