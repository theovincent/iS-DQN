import unittest
import numpy as np
import jax
import jax.numpy as jnp
import optax

from slimdqn.networks.ishldqn import iSHLDQN
from tests.utils import Generator


class TestiSHLDQN(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        self.key = jax.random.PRNGKey(self.random_seed)

        key_actions, key_n_bellman_iterations, key_bins, key_feature_1, key_feature_2, key_feature_3, key_feature_4 = (
            jax.random.split(self.key, 7)
        )
        self.observation_dim = (84, 84, 4)
        self.n_actions = int(jax.random.randint(key_actions, (), minval=2, maxval=10))
        self.n_bellman_iterations = int(jax.random.randint(key_n_bellman_iterations, (), 1, 10))
        self.n_bins = int(jax.random.randint(key_bins, (), minval=2, maxval=10))
        self.q = iSHLDQN(
            self.key,
            self.observation_dim,
            self.n_actions,
            self.n_bellman_iterations,
            self.n_bins,
            [
                jax.random.randint(key_feature_1, (), minval=1, maxval=10),
                jax.random.randint(key_feature_2, (), minval=1, maxval=10),
                jax.random.randint(key_feature_3, (), minval=1, maxval=10),
                jax.random.randint(key_feature_4, (), minval=1, maxval=10),
            ],
            True,
            "impala",
            0.001,
            0.94,
            1,
            1,
            1,
            -10,
            10,
            0.3,
        )

        self.generator = Generator(10, self.observation_dim, self.n_actions)

    def test_compute_target(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        sample = self.generator.sample(self.key)
        idx_network = jax.random.randint(self.key, (), 0, self.n_bellman_iterations)
        # shape (1 + n_bellman_iterations, n_actions, n_bins)
        next_q_logits = self.q.network.apply_fn(self.q.params, sample.next_state)

        computed_target = self.q.compute_target(sample, next_q_logits[idx_network])

        target = sample.reward + (1 - sample.is_terminal) * self.q.gamma * jnp.max(
            jax.nn.softmax(next_q_logits[idx_network]) @ self.q.bin_centers
        )

        self.assertEqual(next_q_logits.shape, (1 + self.n_bellman_iterations, self.n_actions, self.n_bins))
        self.assertEqual(target, computed_target)

    def test_loss(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        samples = self.generator.samples(self.key)

        computed_loss = self.q.loss_on_batch(self.q.params, samples)[0]

        # shape (batch_size, n_bellman_iterations, n_actions, n_bins)
        predictions = self.q.network.apply_fn(self.q.params, samples.state)[:, 1:]
        # shape (batch_size, n_bellman_iterations, n_bins)
        q_logits = jax.vmap(lambda prediction, action: prediction[:, action])(predictions, samples.action)
        # shape (batch_size, n_bellman_iterations, n_actions, n_bins)
        next_q_logits = self.q.network.apply_fn(self.q.params, samples.next_state)[:, :-1]
        targets = jax.vmap(self.q.compute_target)(samples, next_q_logits)
        # shape (batch_size, n_bellman_iterations, n_bins)
        projected_targets = jax.vmap(jax.vmap(self.q.project_target_on_support))(targets)[0]

        loss = optax.softmax_cross_entropy(q_logits, projected_targets).mean(axis=0).sum()

        self.assertEqual(loss, computed_loss)

    def test_best_action(self):
        print(f"-------------- Random key {self.random_seed} --------------")
        state = self.generator.state(self.key)

        computed_best_action = self.q.best_action(self.q.params, state, self.key)

        idx_network = jax.random.randint(self.key, (), 0, self.n_bellman_iterations)
        q_logits = self.q.network.apply_fn(self.q.params, state)[1 + idx_network]
        best_action = jnp.argmax(jax.nn.softmax(q_logits) @ self.q.bin_centers)

        self.assertEqual(q_logits.shape, (self.n_actions, self.n_bins))
        self.assertEqual(best_action, computed_best_action)

    def test_shift_params(self):
        print(f"-------------- Random key {self.random_seed} --------------")
        state = self.generator.state(self.key)
        self.q.params["params"][f"Dense_{self.q.last_idx_mlp}"]["bias"] = (
            jnp.arange((1 + self.n_bellman_iterations) * self.n_actions * self.n_bins) / 100
        )

        # shape (n_bellman_iterations, n_actions)
        q_values = self.q.network.apply_fn(self.q.params, state)
        self.q.params = self.q.shift_params(self.q.params)
        shifted_q_values = self.q.network.apply_fn(self.q.params, state)

        # The target networks are equal to the online networks
        self.assertEqual(jnp.linalg.norm(shifted_q_values[:-1] - q_values[1:]), 0)
