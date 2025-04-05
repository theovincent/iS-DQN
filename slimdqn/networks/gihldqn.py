from typing import Tuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


@jax.jit
def extract_first_params(params):
    return jax.tree.map(lambda param: param[0], params)


@jax.jit
def shift_params(params):
    # Each online network is updated to the following online network
    # \theta_k <- \theta_{k + 1}, i.e., params[k] <- params[k + 1]
    return jax.tree.map(lambda param: param.at[:-1].set(param[1:]), params)


class GIHLDQN:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        n_networks: int,
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
        self.n_networks = n_networks
        self.n_bins = n_bins
        self.network = DQNNet(features, architecture_type, n_actions * self.n_bins)
        self.network.apply_fn = lambda params, state: self.network.apply(params, state).reshape(
            (n_actions, self.n_bins)
        )
        # Create K online parameters
        # params = [\theta_1, \theta_2, ..., \theta_K]
        self.params = jax.vmap(self.network.init, in_axes=(0, None))(
            jax.random.split(key, self.n_networks), jnp.zeros(observation_dim, dtype=jnp.float32)
        )

        # eps_root=1e-9 so that the gradient over adam does not output nans
        self.optimizer = optax.adam(learning_rate, eps=adam_eps, eps_root=1e-9)
        self.optimizer_state = self.optimizer.init(self.params)
        # Create 1 target parameter \bar{\theta}_0
        self.target_params = extract_first_params(self.params)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_frequency = target_update_frequency
        self.cumulated_losses = np.zeros(self.n_networks)
        self.cumulated_unsupported_probs = np.zeros(self.n_networks)
        self.cumulated_variances = np.zeros(self.n_networks)
        self.support = jnp.linspace(min_value, max_value, self.n_bins + 1, dtype=jnp.float32)
        self.bin_centers = (self.support[:-1] + self.support[1:]) / 2
        self.sigma = sigma

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.update_to_data == 0:
            batch_samples = replay_buffer.sample()

            self.params, self.optimizer_state, losses, unsupported_probs, variances = self.learn_on_batch(
                self.params, self.target_params, self.optimizer_state, batch_samples
            )
            self.cumulated_losses += losses
            self.cumulated_unsupported_probs += unsupported_probs
            self.cumulated_variances += variances

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            # The target network is updated to the first online network
            # \bar{\theta}_0 <- \theta_{1}, i.e., target_params <- params[0]
            self.target_params = extract_first_params(self.params)
            # Window shift
            self.params = shift_params(self.params)

            logs = {"loss": self.cumulated_losses.mean() / (self.target_update_frequency / self.update_to_data)}
            for idx_network in range(self.n_networks):
                logs[f"networks/{idx_network}_loss"] = self.cumulated_losses[idx_network] / (
                    self.target_update_frequency / self.update_to_data
                )
                logs[f"networks/{idx_network}_unsupported_prob"] = self.cumulated_unsupported_probs[idx_network] / (
                    self.target_update_frequency / self.update_to_data
                )
                logs[f"networks/{idx_network}_variances"] = self.cumulated_variances[idx_network] / (
                    self.target_update_frequency / self.update_to_data
                )
            self.cumulated_losses = np.zeros_like(self.cumulated_losses)
            self.cumulated_unsupported_probs = np.zeros_like(self.cumulated_unsupported_probs)
            self.cumulated_variances = np.zeros_like(self.cumulated_variances)

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
        # The gradient of each params is composed of 3 terms:
        # Grad_k(||\hat{\Gamma} Q_{k-1} - Q_k||)  # will be given as auxiliary of jax.grad (grad_online_loss)
        # + Grad_k(||\hat{\Gamma} Q_k - Q_{k+1}||) # will be computed by jax.grad (grad_target_variance_loss)
        # + Grad_k(Var(Q_{\theta_{k+1} - lr Grad_{k+1}(||\hat{\Gamma} Q_k - Q_{k+1}||)))
        # will be computed by jax.grad (grad_target_variance_loss)
        grad_target_variance_loss, (grad_online_loss, losses, unsupported_probs, variances) = jax.grad(
            self.loss_on_batch, has_aux=True
        )(params, params_target, batch_samples, optimizer_state)
        updates, optimizer_state = self.optimizer.update(
            jax.tree.map(jnp.add, grad_target_variance_loss, grad_online_loss), optimizer_state
        )
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, losses, unsupported_probs, variances

    def loss_on_batch(self, params: FrozenDict, params_target: FrozenDict, samples: ReplayElement, optimizer_state):
        # Create a list of target params [\bar{\theta}_0, \theta_1, ..., \theta_{K-1}]
        # [target_params, params_0, ..., params_[K - 1]]
        params_targets = jax.tree.map(
            lambda param, target_param: jnp.vstack([target_param[jnp.newaxis], param[:-1]]), params, params_target
        )

        # Stop gradient of the params as loss_on_batch only computes the gradient w.r.t the networks used to compute the targets
        # i.e. the following terms
        # Grad_k(||\hat{\Gamma} Q_k - Q_{k+1}||) + Grad_k(Var(Q_{\theta_{k+1} - lr Grad_{k+1}(||\hat{\Gamma} Q_k - Q_{k+1}||)))
        params = jax.lax.stop_gradient(params)

        # Compute the semi-gradient w.r.t the online networks and update the parameters but not the optimizer state
        # i.e. the following term Grad_k(||\hat{\Gamma} Q_{k-1} - Q_k||)
        grad_loss, (losses, unsupported_probs) = jax.grad(self.vmap_loss, has_aux=True)(params, params_targets, samples)
        updates, _ = self.optimizer.update(grad_loss, optimizer_state)
        updated_params = optax.apply_updates(params, updates)

        # Compute the variance of the online networks, i.e., Var(Q_{\theta_{k+1} - lr Grad_{k+1}(||\hat{\Gamma} Q_k - Q_{k+1}||)
        # map over params, then map over states
        variances = jax.vmap(jax.vmap(self.variance, in_axes=(None, 0)), in_axes=(0, None))(updated_params, samples)

        # Computing the gradient w.r.t params of losses will give Grad_k(||\hat{\Gamma} Q_k - Q_{k+1}||)
        return (losses + variances).mean(axis=1).sum(axis=0), (
            grad_loss,
            losses.mean(axis=1),
            unsupported_probs.mean(axis=1),
            variances.mean(axis=1),
        )

    def vmap_loss(self, params: FrozenDict, params_targets: FrozenDict, samples):
        # map over params, then map over samples
        losses, unsupported_probs = jax.vmap(jax.vmap(self.loss, in_axes=(None, None, 0)), in_axes=(0, 0, None))(
            params, params_targets, samples
        )

        return losses.mean(axis=1).sum(axis=0), (losses, unsupported_probs)

    def loss(self, params: FrozenDict, params_target: FrozenDict, sample: ReplayElement) -> Tuple[jax.Array, jax.Array]:
        # computes the loss for a single sample
        target = self.compute_target(params_target, sample)
        q_logits = self.network.apply_fn(params, sample.state)[sample.action]
        projected_target, unsupported_prob = self.project_target_on_support(target)
        return optax.softmax_cross_entropy(q_logits, projected_target), unsupported_prob

    def compute_target(self, params: FrozenDict, sample: ReplayElement):
        # computes the target value for single sample
        # We first compute the probabilities by applying the softmax on the last axis (bin axis).
        # Then, we compute the expectation by multiplying with the bin centers.
        next_values = jax.nn.softmax(self.network.apply_fn(params, sample.next_state)) @ self.bin_centers
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(next_values)

    def project_target_on_support(self, target: jax.Array) -> jax.Array:
        # We use the error function. It is linked with the cumulative distribution function of a gaussian distribution.
        erf_support = jax.scipy.special.erf((self.support - target) / (jnp.sqrt(2) * self.sigma))
        # We also output the probability mass which does not lie on the support
        # CDF(min support) + 1 - CDF(max support)
        return (
            (erf_support[1:] - erf_support[:-1]) / (erf_support[-1] - erf_support[0] + 1e-6),
            erf_support[0] / 2 + 1 - erf_support[-1] / 2,
        )

    def variance(self, params: FrozenDict, sample: ReplayElement):
        q_probs = jax.nn.softmax(self.network.apply_fn(params, sample.state)[sample.action])

        # V[Q] = E[Q^2] - E[Q]^2
        return q_probs @ self.bin_centers**2 - (q_probs @ self.bin_centers) ** 2

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.Array):
        idx_params = jax.random.randint(key, (), 0, self.n_networks)
        selected_params = jax.tree.map(lambda param: param[idx_params], params)

        # computes the best action for a single state
        # We first compute the probabilities by applying the softmax on the last axis (bin axis).
        # Then, we compute the expectation by multiplying with the bin centers.
        return jnp.argmax(jax.nn.softmax(self.network.apply_fn(selected_params, state)) @ self.bin_centers)

    def get_model(self):
        return {"params": self.params}
