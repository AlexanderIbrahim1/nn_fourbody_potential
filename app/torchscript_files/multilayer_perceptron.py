"""
This module contains a version of the RegressionMultilayerPerceptron with the sizes
specific to the models I plan to make the ensemble potential out of.

The creation of the ordinary RegressionMultilayerPerceptron involves control flow that
doesn't translate well when I want to convert a model from PyTorch to TorchScript. This
version of the RegressionMultilayerPerceptron removes that control flow.
"""

from pathlib import Path
from typing import Union

import torch


class RegressionMultilayerPerceptron(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(6, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def export_torchscript_module(model_filepath: Path) -> None:
    model = RegressionMultilayerPerceptron()
    model.load_state_dict(torch.load(model_filepath))

    example = torch.Tensor([[2.3, 2.4, 2.5, 2.6, 2.7, 2.8]])

    torchscript_module_filepath = model_filepath.with_suffix(".pt")
    torchscript_module = torch.jit.trace(model, example)
    torchscript_module.save(torchscript_module_filepath)


if __name__ == "__main__":
    coarse_energy_model_filepath = Path(".", "models", "coarse_energy_model.pth")
    low_energy_model_filepath = Path(".", "models", "low_energy_model.pth")
    mid_energy_model_filepath = Path(".", "models", "mid_energy_model.pth")
    high_energy_model_filepath = Path(".", "models", "high_energy_model.pth")
    # export_torchscript_module(coarse_energy_model_filepath)
    export_torchscript_module(low_energy_model_filepath)
    export_torchscript_module(mid_energy_model_filepath)
    export_torchscript_module(high_energy_model_filepath)
