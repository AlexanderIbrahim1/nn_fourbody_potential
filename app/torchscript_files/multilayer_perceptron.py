"""
This module contains a version of the RegressionMultilayerPerceptron with the sizes
specific to the models I plan to make the ensemble potential out of.

The creation of the ordinary RegressionMultilayerPerceptron involves control flow that
doesn't translate well when I want to convert a model from PyTorch to TorchScript. This
version of the RegressionMultilayerPerceptron removes that control flow.
"""

import torch


class RegressionMultilayerPerceptron(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self._layers = torch.nn.Sequential(
            [
                torch.nn.Linear(6, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 6),
            ]
        )

    def forward(self, x):
        return self._layers(x)
