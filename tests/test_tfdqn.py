import unittest
import jax
import jax.numpy as jnp
import numpy as np

from slimdqn.networks.tfdqn import TFDQN
from tests.utils import Generator


class TestTFDQN(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        self.key = jax.random.PRNGKey(self.random_seed)

        (key_actions, key_feature_1, key_feature_2, key_feature_3, key_feature_4, key_batch_norm) = jax.random.split(
            self.key, 6
        )
        self.observation_dim = (84, 84, 4)
        self.n_actions = int(jax.random.randint(key_actions, (), minval=2, maxval=10))
        self.q = TFDQN(
            self.key,
            self.observation_dim,
            self.n_actions,
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
            1,
        )

        self.generator = Generator(10, self.observation_dim, self.n_actions)

    def test_compute_target(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        sample = self.generator.sample(self.key)
        q_values_, _ = self.q.network.apply(self.q.params, sample.next_state, mutable=["batch_stats"])
        next_q_values = jnp.squeeze(q_values_)

        computed_target = self.q.compute_target(sample, next_q_values)

        target = sample.reward + (1 - sample.is_terminal) * self.q.gamma * jnp.max(next_q_values)

        self.assertEqual(next_q_values.shape, (self.n_actions,))
        self.assertEqual(target, computed_target)

    def test_loss(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        samples = self.generator.samples(self.key)

        computed_loss = self.q.loss_on_batch(self.q.params, samples)[0]

        # shape (batch_size, n_actions)
        all_q_predictions, _ = self.q.network.apply(
            self.q.params, jnp.concatenate((samples.state, samples.next_state)), mutable=["batch_stats"]
        )
        q_values = jax.vmap(lambda prediction, action: prediction[action])(
            all_q_predictions[: samples.state.shape[0]], samples.action
        )
        # shape (batch_size, 2 * n_bellman_iterations, n_actions)
        targets = jax.vmap(self.q.compute_target)(samples, all_q_predictions[samples.state.shape[0] :])
        loss = jnp.square(q_values - targets).mean(axis=0).sum()

        self.assertEqual(loss, computed_loss)

    def test_best_action(self):
        print(f"-------------- Random key {self.random_seed} --------------")
        state = self.generator.state(self.key)

        computed_best_action = self.q.best_action(self.q.params, state)

        q_values = self.q.network.apply(self.q.params, state, use_running_average=True)
        best_action = jnp.argmax(q_values)

        self.assertEqual(q_values.shape, (self.n_actions,))
        self.assertEqual(best_action, computed_best_action)
