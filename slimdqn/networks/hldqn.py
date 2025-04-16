from functools import partial

import jax
import jax.numpy as jnp
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
        data_to_update: int,
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
        self.data_to_update = data_to_update
        self.target_update_frequency = target_update_frequency
        self.cumulated_loss = 0
        self.cumulated_unsupported_prob = 0
        self.cumulated_entropy = 0
        self.support = jnp.linspace(min_value, max_value, self.n_bins + 1, dtype=jnp.float32)
        self.bin_centers = (self.support[:-1] + self.support[1:]) / 2
        self.clip_target = lambda target: jnp.clip(target, min_value, max_value)
        self.sigma = sigma

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.data_to_update == 0:
            batch_samples = replay_buffer.sample()

            self.params, self.optimizer_state, loss, unsupported_prob, entropy = self.learn_on_batch(
                self.params, self.target_params, self.optimizer_state, batch_samples
            )
            self.cumulated_loss += loss
            self.cumulated_unsupported_prob += unsupported_prob
            self.cumulated_entropy += entropy

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            self.target_params = self.params.copy()

            logs = {
                "loss": self.cumulated_loss / (self.target_update_frequency / self.data_to_update),
                "networks/0_unsupported_prob": self.cumulated_unsupported_prob
                / (self.target_update_frequency / self.data_to_update),
                "entropy": self.cumulated_entropy / (self.target_update_frequency / self.data_to_update),
            }
            self.cumulated_loss = 0
            self.cumulated_unsupported_prob = 0

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
        (loss, unsupported_prob, entropy), grad_loss = jax.value_and_grad(self.loss_on_batch, has_aux=True)(
            params, params_target, batch_samples
        )
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss, unsupported_prob, entropy

    def loss_on_batch(self, params: FrozenDict, params_target: FrozenDict, samples):
        losses, unsupported_probs, entropies = jax.vmap(self.loss, in_axes=(None, None, 0))(
            params, params_target, samples
        )
        return losses.mean(), unsupported_probs.mean(), entropies.mean()

    def loss(self, params: FrozenDict, params_target: FrozenDict, sample: ReplayElement):
        # computes the loss for a single sample
        target = self.compute_target(params_target, sample)
        q_logits = self.network.apply_fn(params, sample.state)[sample.action]
        projected_target, unsupported_prob = self.project_target_on_support(target)
        cross_entropy = optax.softmax_cross_entropy(q_logits, projected_target)
        entropy = -jnp.sum(projected_target * jnp.log(jnp.maximum(projected_target, 1e-5)))
        return cross_entropy, unsupported_prob, entropy

    def compute_target(self, params: FrozenDict, sample: ReplayElement):
        # computes the target value for single sample
        # We first compute the probabilities by applying the softmax on the last axis (bin axis).
        # Then, we compute the expectation by multiplying with the bin centers.
        next_values = jax.nn.softmax(self.network.apply_fn(params, sample.next_state)) @ self.bin_centers
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(next_values)

    def project_target_on_support(self, target: jax.Array) -> jax.Array:
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
