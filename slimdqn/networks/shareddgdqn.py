from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


class SharedDGDQN:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
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
        self.n_actions = n_actions
        self.last_idx_mlp = len(features) if architecture_type == "fc" else len(features) - 3
        self.network = DQNNet(features, architecture_type, 2 * n_actions, layer_norm)
        self.network.apply_fn = lambda params, state: jnp.squeeze(
            self.network.apply(params, state).reshape((-1, 2, n_actions))
        )
        self.params = self.network.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.data_to_update = data_to_update
        self.target_update_frequency = target_update_frequency
        self.cumulated_loss = 0
        self.cumulated_variance = 0

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.data_to_update == 0:
            batch_samples = replay_buffer.sample()

            self.params, self.optimizer_state, loss, variance = self.learn_on_batch(
                self.params, self.optimizer_state, batch_samples
            )
            self.cumulated_loss += loss
            self.cumulated_variance += variance

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            # Window shift
            self.params = self.shift_params(self.params)

            logs = {
                "loss": self.cumulated_loss / (self.target_update_frequency / self.data_to_update),
                "variance": self.cumulated_variance / (self.target_update_frequency / self.data_to_update),
            }
            self.cumulated_loss = 0
            self.cumulated_variance = 0

            return True, logs
        return False, {}

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self,
        params: FrozenDict,
        optimizer_state,
        batch_samples,
    ):
        (loss, variance), grad_loss = jax.value_and_grad(self.loss_on_batch, has_aux=True)(params, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss, variance

    def loss_on_batch(self, params: FrozenDict, samples):
        batch_size = samples.state.shape[0]
        # shape (2 * batch_size, 2, n_actions)
        all_q_values = self.network.apply_fn(params, jnp.concatenate((samples.state, samples.next_state)))
        # shape (batch_size)
        q_values = jax.vmap(lambda q_value, action: q_value[action])(all_q_values[:batch_size, 1], samples.action)
        targets = jax.vmap(self.compute_target)(samples, all_q_values[batch_size:, 0])

        td_losses = jnp.square(q_values - targets)
        variances = jnp.square(targets) - q_values * targets

        return td_losses.mean(), variances.mean()

    def compute_target(self, sample: ReplayElement, next_q_values: jax.Array):
        # shape of next_q_values (next_states, n_actions)
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(
            next_q_values, axis=-1
        )

    @partial(jax.jit, static_argnames="self")
    def shift_params(self, params):
        # Shift the last weight matrix with shape (last_feature, n_networks x n_actions)
        params["params"][f"Dense_{self.last_idx_mlp}"]["kernel"] = (
            params["params"][f"Dense_{self.last_idx_mlp}"]["kernel"]
            .at[:, : -self.n_actions]
            .set(params["params"][f"Dense_{self.last_idx_mlp}"]["kernel"][:, self.n_actions :])
        )
        # Shift the last bias vector with shape (n_networks x n_actions)
        params["params"][f"Dense_{self.last_idx_mlp}"]["bias"] = (
            params["params"][f"Dense_{self.last_idx_mlp}"]["bias"]
            .at[: -self.n_actions]
            .set(params["params"][f"Dense_{self.last_idx_mlp}"]["bias"][self.n_actions :])
        )
        return params

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, **kwargs):
        # computes the best action for a single state
        return jnp.argmax(self.network.apply_fn(params, state)[1])

    def get_model(self):
        return {"params": self.params}
