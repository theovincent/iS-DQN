from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayElement


class iSDQN:
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
        self.network = DQNNet(
            features, architecture_type, (1 + self.n_bellman_iterations) * n_actions, layer_norm, False
        )

        # 1 + self.n_bellman_iterations = [\bar{Q_0}, Q_1, ..., Q_K]
        def apply(params, state):
            q_values = self.network.apply(params, state)
            return q_values.reshape((1 + self.n_bellman_iterations, n_actions))

        self.network.apply_fn = apply
        self.params = self.network.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.data_to_update = data_to_update
        self.target_update_frequency = target_update_frequency
        self.cumulated_losses = np.zeros(self.n_bellman_iterations)

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(self, params: FrozenDict, optimizer_state, batch_samples):
        losses, grad_loss = jax.value_and_grad(self.loss_on_batch)(params, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, losses

    def loss_on_batch(self, params: FrozenDict, samples):
        return jax.vmap(self.loss, in_axes=(None, 0))(params, samples).mean()

    def loss(self, params: FrozenDict, sample):
        # shape (1 + n_bellman_iterations, n_actions)
        q_values = self.network.apply_fn(params, sample.state)[1:, sample.action]
        next_q_values = self.network.apply_fn(params, sample.next_state)
        targets = self.compute_target(sample, next_q_values[:-1])
        stop_grad_targets = jax.lax.stop_gradient(targets)

        return jnp.square(q_values - stop_grad_targets).sum()

    def compute_target(self, sample: ReplayElement, next_q_values: jax.Array):
        # shape of next_q_values (n_bellman_iterations, next_states, n_actions)
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(
            next_q_values, axis=-1
        )

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.Array):
        idx_network = jax.random.randint(key, (), 0, self.n_bellman_iterations)
        q_values = self.network.apply(params, state, use_running_average=True).reshape(
            (1 + self.n_bellman_iterations, self.n_actions)
        )

        # computes the best action for a single state from a uniformly chosen online network
        return jnp.argmax(q_values[1 + idx_network])
