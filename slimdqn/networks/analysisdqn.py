from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


class AnalysisDQN:
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
        self.cumulated_target_churns_train = np.zeros(self.n_bellman_iterations)
        self.cumulated_target_churns_eval = np.zeros(self.n_bellman_iterations)
        self.cumulated_cosine_sim_is_to_tb = 0
        self.cumulated_cosine_sim_tf_to_tb = 0

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.data_to_update == 0:
            batch_samples = replay_buffer.sample()
            batch_samples_eval = replay_buffer.sample()

            (
                self.params,
                self.optimizer_state,
                losses,
                target_churns_train,
                target_churns_eval,
                cosine_sim_is_to_tb,
                cosine_sim_tf_to_tb,
            ) = self.learn_on_batch(
                self.params, self.target_params, self.optimizer_state, batch_samples, batch_samples_eval
            )
            self.cumulated_losses += losses
            self.cumulated_target_churns_train += target_churns_train
            self.cumulated_target_churns_eval += target_churns_eval
            self.cumulated_cosine_sim_is_to_tb += cosine_sim_is_to_tb
            self.cumulated_cosine_sim_tf_to_tb += cosine_sim_tf_to_tb

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            # Window shift
            self.target_params = self.params.copy()
            self.params = self.shift_params(self.params)

            normalizer = self.target_update_frequency / self.data_to_update
            logs = {
                "loss": np.mean(self.cumulated_losses) / normalizer,
                "analysis/target_churns_train": self.cumulated_target_churns_train[0] / normalizer,
                "analysis/target_churns_eval": self.cumulated_target_churns_eval[0] / normalizer,
                "analysis/cosine_sim_iS_to_TB": self.cumulated_cosine_sim_is_to_tb / normalizer,
                "analysis/cosine_sim_TF_to_TB": self.cumulated_cosine_sim_tf_to_tb / normalizer,
            }
            for idx_network in range(min(self.n_bellman_iterations, 5)):
                logs[f"networks/{idx_network}_loss"] = self.cumulated_losses[idx_network] / normalizer
                logs[f"networks/{idx_network}_target_churns_train"] = (
                    self.cumulated_target_churns_train[idx_network] / normalizer
                )
                logs[f"networks/{idx_network}_target_churns_eval"] = (
                    self.cumulated_target_churns_eval[idx_network] / normalizer
                )

            self.cumulated_losses = np.zeros_like(self.cumulated_losses)
            self.cumulated_target_churns_train = np.zeros_like(self.cumulated_target_churns_train)
            self.cumulated_target_churns_eval = np.zeros_like(self.cumulated_target_churns_eval)
            self.cumulated_cosine_sim_is_to_tb = 0
            self.cumulated_cosine_sim_tf_to_tb = 0

            return True, logs

        return False, {}

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(self, params: FrozenDict, params_target, optimizer_state, batch_samples, batch_samples_eval):
        grad_loss, losses, batch_stats, targets_train_pre_update, cosine_sim_is_to_tb, cosine_sim_tf_to_tb = (
            self.grad_and_loss_on_batch(params, params_target, batch_samples)
        )
        all_q_values_eval_pre_udpate, batch_stats = self.network.apply_fn(
            params, jnp.concatenate((batch_samples_eval.state, batch_samples_eval.next_state))
        )
        targets_eval_pre_update = jax.vmap(self.compute_target)(
            batch_samples_eval, all_q_values_eval_pre_udpate[batch_samples_eval.state.shape[0] :, :-1]
        )

        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)
        if self.network.batch_norm:
            params["batch_stats"] = batch_stats["batch_stats"]

        all_q_values_train_post_udpate, batch_stats = self.network.apply_fn(
            params, jnp.concatenate((batch_samples.state, batch_samples.next_state))
        )
        targets_train_post_update = jax.vmap(self.compute_target)(
            batch_samples, all_q_values_train_post_udpate[batch_samples.state.shape[0] :, :-1]
        )
        all_q_values_eval_post_udpate, batch_stats = self.network.apply_fn(
            params, jnp.concatenate((batch_samples_eval.state, batch_samples_eval.next_state))
        )
        targets_eval_post_update = jax.vmap(self.compute_target)(
            batch_samples_eval, all_q_values_eval_post_udpate[batch_samples_eval.state.shape[0] :, :-1]
        )

        return (
            params,
            optimizer_state,
            losses,
            jnp.abs(targets_train_pre_update - targets_train_post_update).mean(axis=0),
            jnp.abs(targets_eval_pre_update - targets_eval_post_update).mean(axis=0),
            cosine_sim_is_to_tb,
            cosine_sim_tf_to_tb,
        )

    def grad_and_loss_on_batch(self, params: FrozenDict, params_target: FrozenDict, samples):
        batch_size = samples.state.shape[0]

        def compute_loss_tb(_params, _params_target, samples):
            q_values, _ = self.network.apply_fn(_params, samples.state)
            next_q_values, _ = self.network.apply_fn(_params_target, samples.next_state)
            targets = jax.vmap(self.compute_target)(
                samples, next_q_values[:, 1]
            )  # first head is used for online and target when full network copied
            td_tb = jax.vmap(lambda q, a: q[a])(q_values[:, 1], samples.action) - jax.lax.stop_gradient(targets)
            return jnp.square(td_tb).mean(axis=0)

        def compute_loss_tf(_params, samples):
            all_q_values, _ = self.network.apply_fn(_params, jnp.concatenate((samples.state, samples.next_state)))
            q_values = jax.vmap(lambda q, a: q[a])(all_q_values[:batch_size, 1], samples.action)
            targets = jax.vmap(self.compute_target)(samples, all_q_values[batch_size:, 1])
            # first head is used for online and target computation in TF
            td_tf = q_values - jax.lax.stop_gradient(targets)
            return jnp.square(td_tf).mean(axis=0)

        def compute_loss_is(_params, samples):
            all_q_values, batch_stats = self.network.apply_fn(
                _params, jnp.concatenate((samples.state, samples.next_state))
            )
            q_values = jax.vmap(lambda q_value, action: q_value[:, action])(
                all_q_values[:batch_size, 1:], samples.action
            )
            targets = jax.vmap(self.compute_target)(samples, all_q_values[batch_size:, :-1])
            td_is = q_values - jax.lax.stop_gradient(targets)
            return jnp.square(td_is).mean(axis=0).sum(), (batch_stats, jnp.square(td_is).mean(axis=0), targets)

        grad_tb = jax.grad(compute_loss_tb)(params, params_target, samples)
        grad_tf = jax.grad(compute_loss_tf)(params, samples)
        grad_is_, (batch_stats, td_losses_is, targets) = jax.grad(compute_loss_is, has_aux=True)(params, samples)

        def extract_feature_gradients(gradients):
            gradients["params"][f"Dense_{self.last_idx_mlp}"]["kernel"] = gradients["params"][
                f"Dense_{self.last_idx_mlp}"
            ]["kernel"][:, self.n_actions : 2 * self.n_actions]
            gradients["params"][f"Dense_{self.last_idx_mlp}"]["bias"] = gradients["params"][
                f"Dense_{self.last_idx_mlp}"
            ]["bias"][self.n_actions : 2 * self.n_actions]
            return jnp.concat(
                [
                    value_grad.reshape(-1)
                    for key_grad, value_grad in sorted(
                        flax.traverse_util.flatten_dict(gradients).items(), key=lambda x: "/".join(x[0])
                    )
                    if not any("norm" in key_part.lower() for key_part in key_grad)
                ]
            )

        grad_tb = extract_feature_gradients(grad_tb)
        grad_tf = extract_feature_gradients(grad_tf)
        grad_is = extract_feature_gradients(jax.tree.map(jnp.copy, grad_is_))

        return (
            grad_is_,
            td_losses_is.sum(),
            batch_stats,
            targets,
            jnp.dot(grad_is, grad_tb) / (jnp.linalg.norm(grad_is) * jnp.linalg.norm(grad_tb) + 1e-9),
            jnp.dot(grad_tf, grad_tb) / (jnp.linalg.norm(grad_tf) * jnp.linalg.norm(grad_tb) + 1e-9),
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
