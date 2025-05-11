from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict
import flax

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


class ExpTFDQN:
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

        self.network.apply_fn = lambda params, state: self.network.apply(params, state, mutable=["batch_stats"])
        self.params = self.network.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.data_to_update = data_to_update
        self.target_update_frequency = target_update_frequency
        self.cumulated_loss = 0
        self.cumulated_q_value_change = 0
        self.cumulated_q_value_abs_change = 0
        self.cumulated_target_change = 0
        self.cumulated_target_abs_change = 0
        self.cumulated_dot_product_target = jax.tree.map(lambda _: 0, self.params)

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.data_to_update == 0:
            batch_samples = replay_buffer.sample()

            (
                self.params,
                self.optimizer_state,
                loss,
                q_value_change,
                q_value_abs_change,
                target_change,
                target_abs_change,
                dot_product_target,
            ) = self.learn_on_batch(self.params, self.optimizer_state, batch_samples)
            self.cumulated_loss += loss
            self.cumulated_q_value_change += q_value_change
            self.cumulated_q_value_abs_change += q_value_abs_change
            self.cumulated_target_change += target_change
            self.cumulated_target_abs_change += target_abs_change
            self.cumulated_dot_product_target = jax.tree.map(
                lambda cumulated_dot_products_w, dot_products_w: cumulated_dot_products_w + dot_products_w,
                self.cumulated_dot_product_target,
                dot_product_target,
            )

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:

            flatten_cumulated_dot_product_target = flax.traverse_util.flatten_dict(
                self.cumulated_dot_product_target["params"], sep="_"
            )

            logs = {
                "loss": self.cumulated_loss / (self.target_update_frequency / self.data_to_update),
            }

            logs[f"analysis/q_value_change"] = self.cumulated_q_value_change / (
                self.target_update_frequency / self.data_to_update
            )
            logs[f"analysis/q_value_abs_change"] = self.cumulated_q_value_abs_change / (
                self.target_update_frequency / self.data_to_update
            )
            logs[f"analysis/target_change"] = self.cumulated_target_change / (
                self.target_update_frequency / self.data_to_update
            )
            logs[f"analysis/target_abs_change"] = self.cumulated_target_abs_change / (
                self.target_update_frequency / self.data_to_update
            )
            logs.update(
                dict(
                    zip(
                        map(
                            lambda key: f"target_sharing/" + key,
                            flatten_cumulated_dot_product_target.keys(),
                        ),
                        map(
                            lambda value: value / (self.target_update_frequency / self.data_to_update),
                            flatten_cumulated_dot_product_target.values(),
                        ),
                    )
                )
            )

            self.cumulated_loss = 0
            self.cumulated_q_value_change = 0
            self.cumulated_q_value_abs_change = 0
            self.cumulated_target_change = 0
            self.cumulated_target_abs_change = 0
            self.cumulated_dot_product_target = jax.tree.map(
                lambda dot_products_w: np.zeros_like(dot_products_w), self.cumulated_dot_product_target
            )

            return True, logs

        return False, {}

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(self, params: FrozenDict, optimizer_state, batch_samples):
        grad_loss, (
            loss,
            batch_stats,
            q_value_change,
            q_value_abs_change,
            target_change,
            target_abs_change,
            dot_product_target,
        ) = jax.grad(self.loss_on_batch, has_aux=True)(params, batch_samples)

        # Compute dot product between the gradient of the loss and the gradient of each term of the loss
        grad_loss, _ = jax.grad(self.loss_on_batch, has_aux=True)(params, batch_samples)

        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)
        if self.network.batch_norm:
            params["batch_stats"] = batch_stats["batch_stats"]

        return (
            params,
            optimizer_state,
            loss,
            q_value_change,
            q_value_abs_change,
            target_change,
            target_abs_change,
            dot_product_target,
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

        td_loss = jnp.square(q_values - stop_grad_targets)

        # Compute changes in q_values and targets
        frozen_all_q_values, _ = self.network.apply_fn(params, jnp.concatenate((samples.state, samples.next_state)))
        frozen_q_values = jax.vmap(lambda q_value, action: q_value[action])(
            frozen_all_q_values[:batch_size], samples.action
        )
        frozen_targets = jax.vmap(self.compute_target)(samples, frozen_all_q_values[batch_size:])
        q_value_change = (q_values - frozen_q_values) / (frozen_q_values + 1e-9)
        q_value_abs_change = jnp.abs(q_value_change)
        target_change = (targets - frozen_targets) / (frozen_targets + 1e-9)
        target_abs_change = jnp.abs(target_change)

        # Compute the dot products between the gradient w.r.t the online network and the gradient w.r.t the target network
        def compute_q_value(params, samples):
            all_q_values_, _ = self.network.apply_fn(params, jnp.concatenate((samples.state, samples.next_state)))
            return jax.vmap(lambda q_value, action: q_value[action])(all_q_values_[:batch_size], samples.action).mean()

        def compute_targets_value(params, samples):
            all_q_values_, _ = self.network.apply_fn(params, jnp.concatenate((samples.state, samples.next_state)))
            return self.compute_target(samples, all_q_values_[batch_size:]).mean()

        grads_q_value = jax.grad(compute_q_value)(params, samples)
        grads_target = jax.grad(compute_targets_value)(params, samples)

        dot_product_target = jax.tree.map(
            lambda grads_q_value_w, grads_target_w: jax.vmap(
                lambda grad_q_value_w, grad_target_w: jnp.dot(
                    grad_q_value_w.flatten(), grad_q_value_w.flatten() - grad_target_w.flatten()
                )
            )(grads_q_value_w, grads_target_w),
            grads_q_value,
            grads_target,
        )

        return td_loss.mean(axis=0), (
            td_loss.mean(axis=0),
            batch_stats,
            q_value_change.mean(axis=0),
            q_value_abs_change.mean(axis=0),
            target_change.mean(axis=0),
            target_abs_change.mean(axis=0),
            dot_product_target,
        )

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
