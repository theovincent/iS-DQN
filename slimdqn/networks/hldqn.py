from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


class HLDQN:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        n_bins: int,
        features: list,
        architecture_type: str,
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_frequency: int,
        min_value: float,
        max_value: float,
        sigma: float,
        adam_eps: float = 1e-8,
    ):
        self.n_bins = n_bins
        self.network = DQNNet(features, architecture_type, n_actions * self.n_bins)
        self.network.apply_fn = lambda params, state: self.network.apply(params, state).reshape(
            (n_actions, self.n_bins)
        )
        self.params = self.network.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)
        self.target_params = self.params

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_frequency = target_update_frequency
        self.cumulated_loss = 0
        self.cumulated_extreme_bins_prob = np.array([0, 0, 0])
        self.support = jnp.linspace(min_value, max_value, self.n_bins + 1, dtype=jnp.float32)
        self.bin_centers = (self.support[:-1] + self.support[1:]) / 2
        self.sigma = sigma

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.update_to_data == 0:
            batch_samples = replay_buffer.sample()

            self.params, self.optimizer_state, loss, extreme_bins_prob = self.learn_on_batch(
                self.params, self.target_params, self.optimizer_state, batch_samples
            )
            self.cumulated_loss += loss
            self.cumulated_extreme_bins_prob += extreme_bins_prob

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            self.target_params = self.params.copy()

            logs = {
                "loss": self.cumulated_loss / (self.target_update_frequency / self.update_to_data),
                "networks/min_bin_prob": self.cumulated_extreme_bins_prob[0]
                / (self.target_update_frequency / self.update_to_data),
                "networks/mean_bin_prob": self.cumulated_extreme_bins_prob[1]
                / (self.target_update_frequency / self.update_to_data),
                "networks/max_bin_prob": self.cumulated_extreme_bins_prob[2]
                / (self.target_update_frequency / self.update_to_data),
            }
            self.cumulated_loss = 0
            self.cumulated_extreme_bins_prob = np.array([0, 0, 0])

            return True, logs
        return False, {}

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self,
        params: FrozenDict,
        params_target: FrozenDict,
        optimizer_state,
        batch_samples,
    ):
        (loss, extreme_bins_prob), grad_loss = jax.value_and_grad(self.loss_on_batch, has_aux=True)(
            params, params_target, batch_samples
        )
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss, extreme_bins_prob

    def loss_on_batch(self, params: FrozenDict, params_target: FrozenDict, samples):
        losses, extreme_bins_prob = jax.vmap(self.loss, in_axes=(None, None, 0))(params, params_target, samples)
        return losses.mean(), extreme_bins_prob.mean(axis=0)

    def loss(self, params: FrozenDict, params_target: FrozenDict, sample: ReplayElement):
        # computes the loss for a single sample
        target = self.compute_target(params_target, sample)
        q_logits = self.network.apply_fn(params, sample.state)[sample.action]
        return (
            optax.softmax_cross_entropy(q_logits, self.project_target_on_support(target)),
            jax.nn.softmax(q_logits)[jnp.array([0, self.n_bins // 2, -1])],
        )

    def compute_target(self, params: FrozenDict, sample: ReplayElement):
        # computes the target value for single sample
        # We first compute the probabilities by applying the softmax on the last axis (bin axis).
        # Then, we compute the expectation by multiplying with the bin centers.
        next_values = jax.nn.softmax(self.network.apply_fn(params, sample.next_state)) @ self.bin_centers
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(next_values)

    def project_target_on_support(self, target: jax.Array) -> jax.Array:
        # We use the error function. It is linked with the cumulative distribution function of a gaussian distribution.
        cdf_evals = jax.scipy.special.erf((self.support - target) / (jnp.sqrt(2) * self.sigma))
        return (cdf_evals[1:] - cdf_evals[:-1]) / (cdf_evals[-1] - cdf_evals[0] + 1e-6)

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, **kwargs):
        # computes the best action for a single state
        # We first compute the probabilities by applying the softmax on the last axis (bin axis).
        # Then, we compute the expectation by multiplying with the bin centers.
        return jnp.argmax(jax.nn.softmax(self.network.apply_fn(params, state)) @ self.bin_centers)

    def get_model(self):
        return {"params": self.params}
