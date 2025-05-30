import unittest
import numpy as np
import jax
import jax.numpy as jnp

from slimdqn.utils.analysis_architecture import AnalysisNet
from slimdqn.utils.analysis import compute_srank, compute_dead_neurons
from tests.utils import Generator


class TestAnalysis(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        self.key = jax.random.PRNGKey(self.random_seed)
        (
            key_layer_norm,
            key_batch_norm,
            key_architecture,
            key_feature_1,
            key_feature_2,
            key_feature_3,
            key_params,
            self.key,
        ) = jax.random.split(self.key, 8)
        self.observation_dim = (84, 84, 4)
        self.q_network = AnalysisNet(
            [
                jax.random.randint(key_feature_1, (), minval=1, maxval=10),
                jax.random.randint(key_feature_2, (), minval=1, maxval=10),
                jax.random.randint(key_feature_3, (), minval=1, maxval=10),
                512,
            ],
            ["cnn", "impala"][jax.random.randint(key_architecture, (), 0, 2)],
            jax.random.uniform(key_layer_norm) > 0.5,
            jax.random.uniform(key_batch_norm) > 0.5,
        )
        self.params = self.q_network.init(key_params, jnp.zeros(self.observation_dim, dtype=jnp.float32))
        self.generator = Generator(51 * 50, self.observation_dim, None)

    def test_srank(self) -> None:
        key_batch_size, key_states, self.key = jax.random.split(self.key, 3)
        batch_size = jax.random.randint(key_batch_size, (), 1, 3000)
        self.assertEqual(compute_srank(jnp.ones((batch_size, 512))), 1)
        self.assertEqual(
            compute_srank(jnp.diag(jnp.arange(0, batch_size))),
            np.searchsorted(
                np.cumsum(jnp.arange(0, batch_size)[::-1]), 0.99 * batch_size * (batch_size - 1) / 2, side="left"
            )
            + 1,
        )

        states = self.generator.states(key_states)
        (feature_matrix, _), _ = self.q_network.apply(self.params, states, mutable=["batch_stats"])

        self.assertGreater(compute_srank(feature_matrix), 256)
        self.assertEqual(compute_srank(feature_matrix, 1), 1)  # for threshold 1, srank=1

    def test_dead_neurons(self) -> None:
        key_states, self.key = jax.random.split(self.key, 2)
        states = self.generator.states(key_states)

        (_, score_neurons), _ = self.q_network.apply(self.params, states, mutable=["batch_stats"])
        self.assertLess(compute_dead_neurons(score_neurons), 0.1)

        for key in self.params["params"].keys():
            if "Conv" in key or "Dense" in key:
                self.params["params"][key]["kernel"] = jnp.zeros_like(self.params["params"][key]["kernel"])
                self.params["params"][key]["bias"] = jnp.zeros_like(self.params["params"][key]["bias"])
            elif "Stack" in key:
                for stack_key in self.params["params"][key].keys():
                    if "Conv" in stack_key or "Dense" in stack_key:
                        self.params["params"][key][stack_key]["kernel"] = jnp.zeros_like(
                            self.params["params"][key][stack_key]["kernel"]
                        )
                        self.params["params"][key][stack_key]["bias"] = jnp.zeros_like(
                            self.params["params"][key][stack_key]["bias"]
                        )

        (_, score_neurons), _ = self.q_network.apply(self.params, states, mutable=["batch_stats"])
        self.assertEqual(compute_dead_neurons(score_neurons), 1)
