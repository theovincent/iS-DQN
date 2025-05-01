import unittest
import numpy as np
import jax
import jax.numpy as jnp

from slimdqn.networks.shareddgdqn import SharedDGDQN
from tests.utils import Generator


class TestSharedDGDQN(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        self.key = jax.random.PRNGKey(self.random_seed)

        key_actions, key_feature_1, key_feature_2, key_feature_3, key_feature_4 = jax.random.split(self.key, 5)
        self.observation_dim = (84, 84, 4)
        self.n_actions = int(jax.random.randint(key_actions, (), minval=2, maxval=10))
        self.q = SharedDGDQN(
            self.key,
            self.observation_dim,
            self.n_actions,
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
        )

        self.generator = Generator(10, self.observation_dim, self.n_actions)

    def test_compute_target(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        sample = self.generator.sample(self.key)

        computed_target = self.q.compute_target(sample, self.q.network.apply_fn(self.q.params, sample.next_state)[0])

        next_q_values = self.q.network.apply_fn(self.q.params, sample.next_state)[0]
        target = sample.reward + (1 - sample.is_terminal) * self.q.gamma * jnp.max(next_q_values)

        self.assertEqual(next_q_values.shape, (self.n_actions,))
        self.assertEqual(target, computed_target)

    def test_loss(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        samples = self.generator.samples(self.key)

        computed_loss = self.q.loss_on_batch(self.q.params, samples)[0]

        # shape (batch_size, n_actions)
        predictions = self.q.network.apply_fn(self.q.params, samples.state)[:, 1]
        q_values = jax.vmap(lambda prediction, action: prediction[action])(predictions, samples.action)
        # shape (batch_size, n_actions)
        next_predictions = self.q.network.apply_fn(self.q.params, samples.next_state)[:, 0]
        targets = samples.reward + (1 - samples.is_terminal) * self.q.gamma * jnp.max(next_predictions, axis=-1)

        loss = jnp.square(q_values - targets).mean()

        self.assertEqual(loss, computed_loss)

    def test_best_action(self):
        print(f"-------------- Random key {self.random_seed} --------------")
        state = self.generator.state(self.key)

        computed_best_action = self.q.best_action(self.q.params, state)

        q_values = self.q.network.apply_fn(self.q.params, state)[1]
        best_action = jnp.argmax(q_values)

        self.assertEqual(q_values.shape, (self.n_actions,))
        self.assertEqual(best_action, computed_best_action)
