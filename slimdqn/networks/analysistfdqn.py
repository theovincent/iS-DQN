from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


class AnalysisTFDQN:
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
        self.n_actions = n_actions
        self.last_idx_mlp = len(features) if architecture_type == "fc" else len(features) - 3
        self.network = DQNNet(features, architecture_type, n_actions, layer_norm, batch_norm)

        self.params = self.network.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))
        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.data_to_update = data_to_update
        self.target_update_frequency = target_update_frequency
        self.cumulated_loss = 0
        self.cumulated_target_churn_train = 0
        self.cumulated_target_churn_eval = 0

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.data_to_update == 0:
            batch_samples = replay_buffer.sample()
            batch_samples_eval = replay_buffer.sample()

            (
                self.params,
                self.optimizer_state,
                loss,
                target_churn_train,
                target_churn_eval,
            ) = self.learn_on_batch(self.params, self.optimizer_state, batch_samples, batch_samples_eval)
            self.cumulated_loss += loss
            self.cumulated_target_churn_train += target_churn_train
            self.cumulated_target_churn_eval += target_churn_eval

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            normalizer = self.target_update_frequency / self.data_to_update
            logs = {
                "loss": self.cumulated_loss / normalizer,
                "analysis/target_churn_train": self.cumulated_target_churn_train / normalizer,
                "analysis/target_churn_eval": self.cumulated_target_churn_eval / normalizer,
            }

            self.cumulated_loss = 0
            self.cumulated_target_churn_train = 0
            self.cumulated_target_churn_eval = 0

            return True, logs

        return False, {}

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(self, params: FrozenDict, optimizer_state, batch_samples, batch_samples_eval):
        (loss, (batch_stats, targets_train_pre_update)), grad_loss = jax.value_and_grad(
            self.loss_on_batch, has_aux=True
        )(params, batch_samples)
        all_q_values_eval_pre_udpate, batch_stats = self.network.apply(
            params, jnp.concatenate((batch_samples_eval.state, batch_samples_eval.next_state)), mutable=["batch_stats"]
        )
        targets_eval_pre_update = self.compute_target(
            batch_samples_eval, all_q_values_eval_pre_udpate[batch_samples_eval.state.shape[0] :]
        )

        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)
        if self.network.batch_norm:
            params["batch_stats"] = batch_stats["batch_stats"]

        all_q_values_train_post_udpate, batch_stats = self.network.apply(
            params, jnp.concatenate((batch_samples.state, batch_samples.next_state)), mutable=["batch_stats"]
        )
        targets_train_post_update = self.compute_target(
            batch_samples, all_q_values_train_post_udpate[batch_samples.state.shape[0] :]
        )
        all_q_values_eval_post_udpate, batch_stats = self.network.apply(
            params, jnp.concatenate((batch_samples_eval.state, batch_samples_eval.next_state)), mutable=["batch_stats"]
        )
        targets_eval_post_update = self.compute_target(
            batch_samples_eval, all_q_values_eval_post_udpate[batch_samples_eval.state.shape[0] :]
        )

        return (
            params,
            optimizer_state,
            loss,
            jnp.abs(targets_train_pre_update - targets_train_post_update).mean(),
            jnp.abs(targets_eval_pre_update - targets_eval_post_update).mean(),
        )

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

        return td_losses.mean(), (batch_stats, targets)

    def compute_target(self, sample: ReplayElement, next_q_values: jax.Array):
        # shape of next_q_values (n_bellman_iterations, next_states, n_actions)
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(
            next_q_values, axis=-1
        )

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, **kwargs):
        # computes the best action for a single state
        return jnp.argmax(self.network.apply(params, state, use_running_average=True))

    def get_model(self):
        return {"params": self.params}
