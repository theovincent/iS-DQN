import numpy as np

from slimdqn.utils.analysis_architecture import AnalysisNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer
from slimdqn.utils.analysis import compute_srank, compute_dead_neurons


def eval_srank_and_dead_neurons(params, rb: ReplayBuffer, p):
    q_network = AnalysisNet(
        p["features"],
        p["architecture_type"],
        p["layer_norm"],
        p["batch_norm"],
    )

    samples = rb.sample(size=2048)  # Typically 2048 used for srank
    (feature_matrix, score_neurons), _ = q_network.apply(params, samples.state, mutable=["batch_stats"])

    return {
        "srank": float(compute_srank(feature_matrix)),
        "dead_neurons": float(compute_dead_neurons(score_neurons)),
    }
