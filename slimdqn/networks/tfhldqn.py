from typing import Tuple
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


class TFHLDQN:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        n_bins: int,
        features: list,
        layer_norm: bool,
        architecture_type: str,
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        data_to_update: int,
        target_update_frequency: int,
        min_value: float,
        max_value: float,
        sigma: float,
        adam_eps: float = 1e-8,
    ):
        self.n_bins = n_bins
        self.network = DQNNet(features, architecture_type, n_actions * self.n_bins, layer_norm)
        self.network.apply_fn = lambda params, state: jnp.squeeze(
            self.network.apply(params, state).reshape((-1, n_actions, self.n_bins))
        )
        self.params = self.network.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.data_to_update = data_to_update
        self.target_update_frequency = target_update_frequency
        self.cumulated_loss = 0
        self.cumulated_unsupported_prob = 0
        self.support = jnp.linspace(min_value, max_value, self.n_bins + 1, dtype=jnp.float32)
        self.bin_centers = (self.support[:-1] + self.support[1:]) / 2
        self.clip_target = lambda target: jnp.clip(target, min_value, max_value)
        self.sigma = sigma

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.data_to_update == 0:
            batch_samples = replay_buffer.sample()

            self.params, self.optimizer_state, loss, unsupported_prob = self.learn_on_batch(
                self.params, self.optimizer_state, batch_samples
            )
            self.cumulated_loss += loss
            self.cumulated_unsupported_prob += unsupported_prob

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            logs = {
                "loss": self.cumulated_loss / (self.target_update_frequency / self.data_to_update),
                "networks/0_unsupported_prob": self.cumulated_unsupported_prob
                / (self.target_update_frequency / self.data_to_update),
            }
            self.cumulated_loss = 0
            self.cumulated_unsupported_prob = 0

            return True, logs
        return False, {}

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self,
        params: FrozenDict,
        optimizer_state,
        batch_samples,
    ):
        (loss, unsupported_prob), grad_loss = jax.value_and_grad(self.loss_on_batch, has_aux=True)(
            params, batch_samples
        )
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss, unsupported_prob

    def loss_on_batch(self, params: FrozenDict, samples):
        batch_size = samples.state.shape[0]
        # shape (2 * batch_size, n_actions, n_bins)
        all_q_logits = self.network.apply_fn(params, jnp.concatenate((samples.state, samples.next_state)))
        # share (batch_size, n_bins)
        q_logits = jax.vmap(lambda q_value, action: q_value[action])(all_q_logits[:batch_size], samples.action)
        # share (batch_size)
        targets = self.compute_target(samples, all_q_logits[batch_size:])
        # share (batch_size, n_bins)
        projected_targets, unsupported_probs = jax.vmap(self.project_target_on_support)(targets)
        stop_grad_projected_targets = jax.lax.stop_gradient(projected_targets)

        # shape (batch_size)
        cross_entropy = optax.softmax_cross_entropy(q_logits, stop_grad_projected_targets)
        return cross_entropy.mean(), unsupported_probs.mean()

    def compute_target(self, sample: ReplayElement, next_q_logits: jax.Array):
        # computes the target value for single sample
        # We first compute the probabilities by applying the softmax on the last axis (bin axis).
        # Then, we compute the expectation by multiplying with the bin centers.
        next_values = jax.nn.softmax(next_q_logits) @ self.bin_centers
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(
            next_values, axis=-1
        )

    def project_target_on_support(self, target: jax.Array) -> Tuple[jax.Array, jax.Array]:
        # We use the error function. It is linked with the cumulative distribution function of a gaussian distribution.
        erf_support = jax.scipy.special.erf((self.support - self.clip_target(target)) / (jnp.sqrt(2) * self.sigma))
        # We also output the probability mass which does not lie on the support
        # CDF(min support) + 1 - CDF(max support)
        return (
            (erf_support[1:] - erf_support[:-1]) / (erf_support[-1] - erf_support[0] + 1e-9),
            erf_support[0] / 2 + 1 - erf_support[-1] / 2,
        )

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, **kwargs):
        # computes the best action for a single state
        # We first compute the probabilities by applying the softmax on the last axis (bin axis).
        # Then, we compute the expectation by multiplying with the bin centers.
        return jnp.argmax(jax.nn.softmax(self.network.apply_fn(params, state)) @ self.bin_centers)

    def get_model(self):
        return {"params": self.params}
