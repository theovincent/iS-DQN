import unittest
import numpy as np
import jax
import jax.numpy as jnp

from slimdqn.networks.sharedgidqn import SharedGIDQN
from tests.utils import Generator


class TestSharedGIDQN(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = 288  # np.random.randint(1000)
        self.key = jax.random.PRNGKey(self.random_seed)

        key_actions, key_n_networks, key_feature_1, key_feature_2, key_feature_3, key_feature_4 = jax.random.split(
            self.key, 6
        )
        self.observation_dim = (84, 84, 4)
        self.n_actions = int(jax.random.randint(key_actions, (), minval=2, maxval=10))
        self.n_networks = int(jax.random.randint(key_n_networks, (), 2, 10))
        self.q = SharedGIDQN(
            self.key,
            self.observation_dim,
            self.n_actions,
            self.n_networks,
            [
                jax.random.randint(key_feature_1, (), minval=5, maxval=20),
                jax.random.randint(key_feature_2, (), minval=5, maxval=20),
                jax.random.randint(key_feature_3, (), minval=5, maxval=20),
                jax.random.randint(key_feature_4, (), minval=5, maxval=20),
            ],
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
        idx_params = jax.random.randint(self.key, (), 0, self.n_networks)

        computed_target = self.q.compute_target(sample, self.q.network.apply_fn(self.q.params, sample.next_state))[
            idx_params
        ]

        next_q_values = self.q.network.apply_fn(self.q.params, sample.next_state)[idx_params]
        target = sample.reward + (1 - sample.is_terminal) * self.q.gamma * jnp.max(next_q_values)

        self.assertEqual(next_q_values.shape, (self.n_actions,))
        self.assertEqual(target, computed_target)

    def test_loss(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        samples = self.generator.samples(self.key)

        computed_loss = self.q.loss_on_batch(self.q.params, samples)[0]

        # shape (batch_size, n_networks, n_actions)
        predictions = self.q.network.apply_fn(self.q.params, samples.state)
        q_values = jax.vmap(lambda prediction, action: prediction[:, action])(predictions, samples.action)
        loss = jnp.square(q_values[:, 1:]).mean(axis=0).sum()

        self.assertEqual(loss, computed_loss)

    def test_best_action(self):
        print(f"-------------- Random key {self.random_seed} --------------")
        state = self.generator.state(self.key)

        computed_best_action = self.q.best_action(self.q.params, state, self.key)

        idx_params = jax.random.randint(self.key, (), 0, self.n_networks)
        q_values = self.q.network.apply_fn(self.q.params, state)[idx_params]
        best_action = jnp.argmax(q_values)

        self.assertEqual(q_values.shape, (self.n_actions,))
        self.assertEqual(best_action, computed_best_action)

    def test_shift_params(self):
        print(f"-------------- Random key {self.random_seed} --------------")
        state = self.generator.state(self.key)
        self.q.params["params"][f"Dense_{self.q.last_idx_mlp}"]["bias"] = (
            jnp.arange(self.n_networks * self.n_actions) / 100
        )

        # shape (n_networks, n_actions)
        q_values = self.q.network.apply_fn(self.q.params, state)
        self.q.params = self.q.shift_params(self.q.params)
        shifted_q_values = self.q.network.apply_fn(self.q.params, state)

        self.assertEqual(jnp.linalg.norm(shifted_q_values[:-1] - q_values[1:]), 0)
        self.assertEqual(jnp.linalg.norm(shifted_q_values[-1] - shifted_q_values[-1]), 0)
