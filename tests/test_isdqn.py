import unittest
import jax
import jax.numpy as jnp
import numpy as np

from slimdqn.networks.isdqn import iSDQN
from tests.utils import Generator


class TestiSDQN(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        self.key = jax.random.PRNGKey(self.random_seed)

        (
            key_actions,
            key_n_bellman_iterations,
            key_feature_1,
            key_feature_2,
            key_feature_3,
            key_feature_4,
            key_batch_norm,
        ) = jax.random.split(self.key, 7)
        self.observation_dim = (84, 84, 4)
        self.n_actions = int(jax.random.randint(key_actions, (), minval=2, maxval=10))
        self.n_bellman_iterations = int(jax.random.randint(key_n_bellman_iterations, (), 1, 10))
        self.q = iSDQN(
            self.key,
            self.observation_dim,
            self.n_actions,
            self.n_bellman_iterations,
            [
                jax.random.randint(key_feature_1, (), minval=5, maxval=20),
                jax.random.randint(key_feature_2, (), minval=5, maxval=20),
                jax.random.randint(key_feature_3, (), minval=5, maxval=20),
                jax.random.randint(key_feature_4, (), minval=5, maxval=20),
            ],
            True,
            jax.random.uniform(key_batch_norm) > 0.5,
            "cnn",
            0.001,
            0.94,
            1,
            1,
            1,
        )

        self.generator = Generator(10, self.observation_dim, self.n_actions)

    def test_compute_target(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        sample = self.generator.sample(self.key)
        idx_network = jax.random.randint(self.key, (), 0, self.n_bellman_iterations)
        q_values_, _ = self.q.network.apply_fn(self.q.params, sample.next_state)
        next_q_values = jnp.squeeze(q_values_)[idx_network]

        computed_target = self.q.compute_target(sample, next_q_values)

        target = sample.reward + (1 - sample.is_terminal) * self.q.gamma * jnp.max(next_q_values)

        self.assertEqual(next_q_values.shape, (self.n_actions,))
        self.assertEqual(target, computed_target)

    def test_loss(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        samples = self.generator.samples(self.key)

        computed_loss = self.q.loss_on_batch(self.q.params, samples)[0]

        # shape (batch_size, 1 + n_bellman_iterations, n_actions)
        all_q_predictions, _ = self.q.network.apply_fn(
            self.q.params, jnp.concatenate((samples.state, samples.next_state))
        )
        q_values = jax.vmap(lambda prediction, action: prediction[:, action])(
            all_q_predictions[: samples.state.shape[0], 1:], samples.action
        )
        # shape (batch_size, 1 + n_bellman_iterations, n_actions)
        targets = jax.vmap(self.q.compute_target)(samples, all_q_predictions[samples.state.shape[0] :, :-1])
        loss = jnp.square(q_values - targets).mean(axis=0).sum()

        self.assertEqual(loss, computed_loss)

    def test_best_action(self):
        print(f"-------------- Random key {self.random_seed} --------------")
        state = self.generator.state(self.key)

        computed_best_action = self.q.best_action(self.q.params, state, self.key)

        idx_network = jax.random.randint(self.key, (), 0, self.n_bellman_iterations)
        q_values = self.q.network.apply(self.q.params, state, use_running_average=True).reshape(
            (1 + self.n_bellman_iterations, self.n_actions)
        )[idx_network]
        best_action = jnp.argmax(q_values)

        self.assertEqual(q_values.shape, (self.n_actions,))
        self.assertEqual(best_action, computed_best_action)

    def test_shift_params(self):
        print(f"-------------- Random key {self.random_seed} --------------")
        state = self.generator.state(self.key)
        self.q.params["params"][f"Dense_{self.q.last_idx_mlp}"]["bias"] = (
            jnp.arange((1 + self.n_bellman_iterations) * self.n_actions) / 100
        )

        # shape (n_bellman_iterations, n_actions)
        q_values = self.q.network.apply(self.q.params, state, use_running_average=True).reshape(
            (1 + self.n_bellman_iterations, self.n_actions)
        )
        self.q.params = self.q.shift_params(self.q.params)
        shifted_q_values = self.q.network.apply(self.q.params, state, use_running_average=True).reshape(
            (1 + self.n_bellman_iterations, self.n_actions)
        )

        # The target networks are equal to the online networks
        self.assertEqual(jnp.linalg.norm(shifted_q_values[:-1] - q_values[1:]), 0)
        # The online networks have not changed
        self.assertEqual(jnp.linalg.norm(shifted_q_values[1:] - q_values[1:]), 0)
