"""
This module contains the neural network model to be trained.

TODO: turn asserts into exceptions
"""

import torch

from torchtyping import TensorType
from torchtyping import patch_typeguard
from typeguard import typechecked


patch_typeguard()


@typechecked
class RegressionMultilayerPerceptron(torch.nn.Module):
    def __init__(
        self, n_features: int, n_outputs: int, hidden_layer_sizes: list[int]
    ) -> None:
        super().__init__()

        self._layer_sizes = [n_features] + hidden_layer_sizes + [n_outputs]
        self.layers = _create_linear_sequential(self._layer_sizes)

    def forward(
        self, x: TensorType["batch", "features"]
    ) -> TensorType["batch", "outputs"]:
        return self.layers(x)

    @property
    def n_features(self) -> int:
        return self._layer_sizes[0]

    @property
    def n_outputs(self) -> int:
        return self._layer_sizes[-1]

    @property
    def layer_sizes(self) -> list[int]:
        return self._layer_sizes


def _create_linear_sequential(layer_sizes: list[int]) -> torch.nn.Sequential:
    """
    Create a sequence of perceptrons with ReLU activation functions between them.

    A future change could involve allowing more than just ReLU?
    """
    n_sizes = len(layer_sizes)

    assert n_sizes >= 2
    for size in layer_sizes:
        assert size >= 1

    if n_sizes == 2:
        layers = [torch.nn.Linear(layer_sizes[0], layer_sizes[1])]
    else:
        layers = []
        n_pairs = n_sizes - 1

        # the last pair of sizes is a special case
        for i in range(n_pairs - 1):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(torch.nn.ReLU())

        # the last pair of sizes
        layers.append(torch.nn.Linear(layer_sizes[-2], layer_sizes[-1]))

    return torch.nn.Sequential(*layers)
