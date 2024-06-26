"""
This module contains the neural network model to be trained.
"""

from typing import Callable

import torch

from nn_fourbody_potential.transformations.transformers import SixSideLengthsTransformer


# dataclass in Python 3.9 doesn't have support for keyword-only parameters, so I have to
# write out the entire constructor
class TrainingParameters:
    def __init__(
        self,
        seed: int,
        layers: list[int],
        learning_rate: float,
        weight_decay: float,
        training_size: int,
        total_epochs: int,
        batch_size: int,
        transformations: list[SixSideLengthsTransformer],
        apply_batch_norm: bool,
        other: str,
    ) -> None:
        self.seed = seed
        self.layers = layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.training_size = training_size
        self.total_epochs = total_epochs
        self.batch_size = batch_size
        self.transformations = transformations
        self.apply_batch_norm = apply_batch_norm
        self.other = other


class RegressionMultilayerPerceptron(torch.nn.Module):
    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        hidden_layer_sizes: list[int],
        apply_batch_norm: bool = False,
        *,
        activation_function_factory: Callable[[], torch.nn.Module] = torch.nn.ReLU,
    ) -> None:
        super().__init__()

        self._layer_sizes = [n_features] + hidden_layer_sizes + [n_outputs]
        self.layers = _create_linear_sequential(self._layer_sizes, apply_batch_norm, activation_function_factory)

    def forward(self, x):
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


def _create_linear_sequential(
    layer_sizes: list[int],
    apply_batch_norm: bool,
    activation_function_factory: Callable[[], torch.nn.Module],
) -> torch.nn.Sequential:
    """
    Create a sequence of perceptrons with ReLU activation functions between them.

    A future change could involve allowing more than just ReLU?
    """
    n_sizes = len(layer_sizes)

    if n_sizes < 2:
        raise ValueError(
            "Need at least two layers (the input and output layers)\n" f"Found number of layers: {n_sizes}"
        )

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
            if apply_batch_norm:
                layers.append(torch.nn.BatchNorm1d(num_features=layer_sizes[i + 1]))
            layers.append(activation_function_factory())

        # the last pair of sizes
        layers.append(torch.nn.Linear(layer_sizes[-2], layer_sizes[-1]))

    return torch.nn.Sequential(*layers)
