from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


class CrownGIDQN:
    def __init__(
        self,
        key: jax.Array,
        observation_dim,
        n_actions,
        n_networks: int,
        features: list,
        architecture_type: str,
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        data_to_update: int,
        target_update_frequency: int,
        adam_eps: float = 1e-8,
    ):
        self.n_networks = n_networks
        self.n_actions = n_actions
        self.last_idx_mlp = len(features) if architecture_type == "fc" else len(features) - 3
        self.network = DQNNet(features, architecture_type, self.n_networks * n_actions)
        self.network.apply_fn = lambda params, state: jnp.squeeze(
            self.network.apply(params, state).reshape((-1, self.n_networks, n_actions))
        )
        self.params = self.network.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.data_to_update = data_to_update
        self.target_update_frequency = target_update_frequency
        self.cumulated_losses = np.zeros(self.n_networks)
        self.cumulated_variances = np.zeros(self.n_networks)

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.data_to_update == 0:
            batch_samples = replay_buffer.sample()

            self.params, self.optimizer_state, losses, variances = self.learn_on_batch(
                self.params, self.optimizer_state, batch_samples
            )
            self.cumulated_losses += losses
            self.cumulated_variances += variances

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            logs = {
                "loss": np.mean(self.cumulated_losses) / (self.target_update_frequency / self.data_to_update),
                "variance": np.mean(self.cumulated_variances) / (self.target_update_frequency / self.data_to_update),
            }
            for idx_network in range(min(self.n_networks, 5)):
                logs[f"networks/{idx_network}_loss"] = self.cumulated_losses[idx_network] / (
                    self.target_update_frequency / self.data_to_update
                )
                logs[f"networks/{idx_network}_variance"] = self.cumulated_variances[idx_network] / (
                    self.target_update_frequency / self.data_to_update
                )
            self.cumulated_losses = np.zeros_like(self.cumulated_losses)
            self.cumulated_variances = np.zeros_like(self.cumulated_variances)

            return True, logs

        return False, {}

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self,
        params: FrozenDict,
        optimizer_state,
        batch_samples,
    ):
        grad_loss, (losses, variances) = jax.grad(self.loss_on_batch, has_aux=True)(params, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, losses, variances

    def loss_on_batch(self, params: FrozenDict, samples: ReplayElement):
        batch_size = samples.state.shape[0]
        # shape (2 * batch_size, n_networks, n_actions)
        all_q_values = self.network.apply_fn(params, jnp.concatenate((samples.state, samples.next_state)))
        # shape (batch_size, n_networks)
        q_values = jax.vmap(lambda q_value, action: q_value[:, action])(all_q_values[:batch_size], samples.action)
        targets = jax.vmap(self.compute_target)(samples, all_q_values[batch_size:])
        rollded_targets = jnp.roll(targets, shift=1, axis=1)

        # shape (batch_size, n_networks, n_actions)
        losses = jnp.square(q_values) + 2 * rollded_targets * (jax.lax.stop_gradient(q_values) - q_values)
        td_losses = jnp.square(q_values - rollded_targets)
        variances = jnp.square(targets) - q_values * rollded_targets

        return losses.mean(axis=0).sum(), (td_losses.mean(axis=0), variances.mean(axis=0))

    def compute_target(self, sample: ReplayElement, next_q_values: jax.Array):
        # shape of next_q_values (next_states, n_networks, n_actions)
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(
            next_q_values, axis=-1
        )

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.Array):
        idx_params = jax.random.randint(key, (), 0, self.n_networks)

        # computes the best action for a single state
        return jnp.argmax(self.network.apply_fn(params, state)[idx_params])

    def get_model(self):
        return {"params": self.params}
