from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict
import flax

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
        batch_norm: bool,
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
            features, architecture_type, (1 + self.n_bellman_iterations) * n_actions, layer_norm, batch_norm
        )

        # 1 + self.n_bellman_iterations = [\bar{Q_0}, Q_1, ..., Q_K]
        def apply(params, state):
            q_values, batch_stats = self.network.apply(params, state, mutable=["batch_stats"])
            return q_values.reshape((-1, 1 + self.n_bellman_iterations, n_actions)), batch_stats

        self.network.apply_fn = apply
        self.params = self.network.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)
        self.target_params = self.params.copy()

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.data_to_update = data_to_update
        self.target_update_frequency = target_update_frequency
        self.cumulated_losses = np.zeros(self.n_bellman_iterations)
        self.cumulated_q_values_change = np.zeros(self.n_bellman_iterations)
        self.cumulated_q_values_abs_change = np.zeros(self.n_bellman_iterations)
        self.cumulated_targets_change = np.zeros(self.n_bellman_iterations)
        self.cumulated_targets_abs_change = np.zeros(self.n_bellman_iterations)
        self.cumulated_tds_change = np.zeros(self.n_bellman_iterations)
        self.cumulated_q_values_drift = np.zeros(self.n_bellman_iterations)
        self.cumulated_dot_products_targets = jax.tree.map(lambda _: np.zeros(self.n_bellman_iterations), self.params)

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.data_to_update == 0:
            batch_samples = replay_buffer.sample()

            (
                self.params,
                self.optimizer_state,
                losses,
                q_values_change,
                q_values_abs_change,
                targets_change,
                targets_abs_change,
                tds_change,
                q_values_drift,
                dot_products_targets,
            ) = self.learn_on_batch(self.params, self.target_params, self.optimizer_state, batch_samples)
            self.cumulated_losses += losses
            self.cumulated_q_values_change += q_values_change
            self.cumulated_q_values_abs_change += q_values_abs_change
            self.cumulated_targets_change += targets_change
            self.cumulated_targets_abs_change += targets_abs_change
            self.cumulated_tds_change += tds_change
            self.cumulated_q_values_drift += q_values_drift
            self.cumulated_dot_products_targets = jax.tree.map(
                lambda cumulated_dot_products_w, dot_products_w: cumulated_dot_products_w + dot_products_w,
                self.cumulated_dot_products_targets,
                dot_products_targets,
            )

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            # Window shift
            self.target_params = self.params.copy()
            self.params = self.shift_params(self.params)

            flatten_cumulated_dot_products_targets = flax.traverse_util.flatten_dict(
                self.cumulated_dot_products_targets["params"], sep="_"
            )
            normalizer = self.target_update_frequency / self.data_to_update
            logs = {"loss": np.mean(self.cumulated_losses) / normalizer}
            for idx_network in range(min(self.n_bellman_iterations, 5)):
                logs.update(
                    {
                        f"networks/{idx_network}_loss": self.cumulated_losses[idx_network] / normalizer,
                        f"analysis/{idx_network}_q_value_change": self.cumulated_q_values_change[idx_network]
                        / normalizer,
                        f"analysis/{idx_network}_q_value_abs_change": self.cumulated_q_values_abs_change[idx_network]
                        / normalizer,
                        f"analysis/{idx_network}_target_change": self.cumulated_targets_change[idx_network]
                        / normalizer,
                        f"analysis/{idx_network}_target_abs_change": self.cumulated_targets_abs_change[idx_network]
                        / normalizer,
                        f"analysis/{idx_network}_td_change_change": self.cumulated_tds_change[idx_network] / normalizer,
                        f"analysis/{idx_network}_q_value_shift": self.cumulated_q_values_drift[idx_network]
                        / normalizer,
                    }
                )
                logs.update(
                    dict(
                        zip(
                            map(
                                lambda key: f"target_sharing/{idx_network}_{key}",
                                flatten_cumulated_dot_products_targets.keys(),
                            ),
                            map(
                                lambda value: value[idx_network] / normalizer,
                                flatten_cumulated_dot_products_targets.values(),
                            ),
                        )
                    )
                )

            self.cumulated_losses = np.zeros_like(self.cumulated_losses)
            self.cumulated_q_values_change = np.zeros_like(self.cumulated_q_values_change)
            self.cumulated_q_values_abs_change = np.zeros_like(self.cumulated_q_values_abs_change)
            self.cumulated_targets_change = np.zeros_like(self.cumulated_targets_change)
            self.cumulated_targets_abs_change = np.zeros_like(self.cumulated_targets_abs_change)
            self.cumulated_tds_change = np.zeros_like(self.cumulated_tds_change)
            self.cumulated_q_values_drift = np.zeros_like(self.cumulated_q_values_drift)
            self.cumulated_dot_product_targets = jax.tree.map(
                lambda dot_products_w: np.zeros_like(dot_products_w), self.cumulated_dot_product_targets
            )

            return True, logs

        return False, {}

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(self, params: FrozenDict, params_target: FrozenDict, optimizer_state, batch_samples):
        grad_loss, (
            losses,
            batch_stats,
            q_values_change,
            q_values_abs_change,
            targets_change,
            targets_abs_change,
            td_change,
            q_value_drift,
            dot_products_targets,
        ) = jax.grad(self.loss_on_batch, has_aux=True)(params, params_target, batch_samples)

        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)
        if self.network.batch_norm:
            params["batch_stats"] = batch_stats["batch_stats"]

        return (
            params,
            optimizer_state,
            losses,
            q_values_change,
            q_values_abs_change,
            targets_change,
            targets_abs_change,
            td_change,
            q_value_drift,
            dot_products_targets,
        )

    def loss_on_batch(self, params: FrozenDict, params_target: FrozenDict, samples):
        batch_size = samples.state.shape[0]
        # shape (2 * batch_size, 1 + n_bellman_iterations, n_actions) | Dict
        all_q_values, batch_stats = self.network.apply_fn(params, jnp.concatenate((samples.state, samples.next_state)))
        # shape (batch_size, n_bellman_iterations)
        q_values = jax.vmap(lambda q_value, action: q_value[:, action])(all_q_values[:batch_size, 1:], samples.action)
        targets = jax.vmap(self.compute_target)(samples, all_q_values[batch_size:, :-1])
        stop_grad_targets = jax.lax.stop_gradient(targets)

        # shape (batch_size, n_bellman_iterations)
        td_losses = jnp.square(q_values - stop_grad_targets)

        # Compute frozen q_values and targets
        q_values_target = jax.vmap(lambda q_value, action: q_value[:, action])(
            all_q_values[:batch_size, :-1], samples.action
        )
        frozen_all_q_values, _ = self.network.apply_fn(
            params_target, jnp.concatenate((samples.state, samples.next_state))
        )
        frozen_q_values = jax.vmap(lambda q_value, action: q_value[:, action])(
            frozen_all_q_values[:batch_size, 1:], samples.action
        )
        frozen_targets = jax.vmap(self.compute_target)(samples, frozen_all_q_values[batch_size:, :-1])
        frozen_q_values_target = jax.vmap(lambda q_value, action: q_value[:, action])(
            frozen_all_q_values[:batch_size, :-1], samples.action
        )

        # Compute metrics
        q_values_change = (q_values - frozen_q_values) / (frozen_q_values + 1e-9)
        q_values_abs_change = jnp.abs(q_values_change)
        targets_change = (targets - frozen_targets) / (frozen_targets + 1e-9)
        targets_abs_change = jnp.abs(targets_change)
        tds_change = td_losses / (jnp.square(q_values - frozen_targets) + 1e-9)
        q_values_drift = (q_values - q_values_target) / (q_values - frozen_q_values_target + 1e-9)

        # Compute the dot products between the gradient w.r.t the online network and the gradient w.r.t the target network
        def compute_q_value(params, samples, idx_bellman_iteration):
            all_q_values_, _ = self.network.apply_fn(params, jnp.concatenate((samples.state, samples.next_state)))
            return jax.vmap(lambda q_value, action: q_value[action])(
                all_q_values_[:batch_size, 1 + idx_bellman_iteration], samples.action
            ).mean()

        def compute_targets_value(params, samples, idx_bellman_iteration):
            all_q_values_, _ = self.network.apply_fn(params, jnp.concatenate((samples.state, samples.next_state)))
            return self.compute_target(samples, all_q_values_[batch_size:, idx_bellman_iteration]).mean()

        grads_q_value = jax.vmap(jax.grad(compute_q_value), in_axes=(None, None, 0))(
            params, samples, jnp.arange(self.n_bellman_iterations)
        )
        grads_target = jax.vmap(jax.grad(compute_targets_value), in_axes=(None, None, 0))(
            params, samples, jnp.arange(self.n_bellman_iterations)
        )
        dot_products_targets = jax.tree.map(
            lambda grads_q_value_w, grads_target_w: jax.vmap(
                lambda grad_q_value_w, grad_target_w: jnp.dot(
                    grad_q_value_w.flatten(), grad_q_value_w.flatten() - grad_target_w.flatten()
                )
            )(grads_q_value_w, grads_target_w),
            grads_q_value,
            grads_target,
        )

        return td_losses.mean(axis=0).sum(), (
            td_losses.mean(axis=0),
            batch_stats,
            q_values_change.mean(axis=0),
            q_values_abs_change.mean(axis=0),
            targets_change.mean(axis=0),
            targets_abs_change.mean(axis=0),
            tds_change.mean(axis=0),
            q_values_drift.mean(axis=0),
            dot_products_targets,
        )

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
        idx_network = jax.random.randint(key, (), 0, self.n_bellman_iterations)
        q_values = self.network.apply(params, state, use_running_average=True).reshape(
            (1 + self.n_bellman_iterations, self.n_actions)
        )

        # computes the best action for a single state from a uniformly chosen online network
        return jnp.argmax(q_values[1 + idx_network])

    def get_model(self):
        return {"params": self.params}
