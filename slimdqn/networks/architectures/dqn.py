from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp


class Stack(nn.Module):
    """Stack of pooling and convolutional blocks with residual connections."""

    stack_size: int
    layer_norm: bool
    batch_norm: bool

    @nn.compact
    def __call__(self, x, use_running_average=False):
        initializer = nn.initializers.xavier_uniform()
        x = nn.Conv(
            features=self.stack_size,
            kernel_size=(3, 3),
            kernel_init=initializer,
        )(x)
        x = nn.max_pool(x, window_shape=(3, 3), padding="SAME", strides=(2, 2))

        for _ in range(2):
            block_input = x
            if self.layer_norm:
                x = nn.LayerNorm(reduction_axes=(1, 2, 3))(x)
            elif self.batch_norm:
                x = nn.BatchNorm(use_running_average, axis=(1, 2))(x)
            x = nn.Conv(features=self.stack_size, kernel_size=(3, 3))(nn.relu(x))
            x = nn.Conv(features=self.stack_size, kernel_size=(3, 3))(nn.relu(x))
            x += block_input

        return x


class DQNNet(nn.Module):
    features: Sequence[int]
    architecture_type: str
    final_feature: int
    layer_norm: bool = False
    batch_norm: bool = False

    @nn.compact
    def __call__(self, x, use_running_average=False):
        if self.architecture_type == "cnn":
            initializer = nn.initializers.xavier_uniform()
            idx_feature_start = 3
            x = nn.Conv(features=self.features[0], kernel_size=(8, 8), strides=(4, 4), kernel_init=initializer)(
                jnp.array(x, ndmin=4) / 255.0
            )
            if self.layer_norm:
                x = nn.LayerNorm(reduction_axes=(1, 2, 3))(x)
            elif self.batch_norm:
                x = nn.BatchNorm(use_running_average, axis=(1, 2))(x)
            x = nn.Conv(features=self.features[1], kernel_size=(4, 4), strides=(2, 2), kernel_init=initializer)(
                nn.relu(x)
            )
            if self.layer_norm:
                x = nn.LayerNorm(reduction_axes=(1, 2, 3))(x)
            elif self.batch_norm:
                x = nn.BatchNorm(use_running_average, axis=(1, 2))(x)
            x = nn.Conv(features=self.features[2], kernel_size=(3, 3), strides=(1, 1), kernel_init=initializer)(
                nn.relu(x)
            )
            if self.layer_norm:
                x = nn.LayerNorm(reduction_axes=(1, 2, 3))(x)
            elif self.batch_norm:
                x = nn.BatchNorm(use_running_average, axis=(1, 2))(x)
            x = nn.relu(x).reshape((x.shape[0], -1))
        elif self.architecture_type == "impala":
            initializer = nn.initializers.xavier_uniform()
            idx_feature_start = 3
            x = Stack(self.features[0], self.layer_norm, self.batch_norm)(jnp.array(x, ndmin=4) / 255.0)
            x = Stack(self.features[1], self.layer_norm, self.batch_norm)(x)
            x = Stack(self.features[2], self.layer_norm, self.batch_norm)(x)
            if self.layer_norm:
                x = nn.LayerNorm(reduction_axes=(1, 2, 3))(x)
            elif self.batch_norm:
                x = nn.BatchNorm(use_running_average, axis=(1, 2))(x)
            x = nn.relu(x).reshape((x.shape[0], -1))
        elif self.architecture_type == "fc":
            initializer = nn.initializers.lecun_normal()
            idx_feature_start = 0

        x = jnp.squeeze(x)

        for idx_layer in range(idx_feature_start, len(self.features)):
            x = nn.Dense(self.features[idx_layer], kernel_init=initializer)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            elif self.batch_norm:
                x = nn.BatchNorm(use_running_average)(x)
            x = nn.relu(x)

        return nn.Dense(self.final_feature, kernel_init=initializer)(x)
