from dataclasses import dataclass
from pathlib import Path

import torch

from nn_fourbody_potential.models import RegressionMultilayerPerceptron
from nn_fourbody_potential.modelio.utils import get_model_filename


@dataclass
class CheckpointData:
    model: RegressionMultilayerPerceptron
    optimizer: torch.optim.Optimizer
    epoch: int


@dataclass(frozen=True)
class CheckpointLoader:
    """Loads objects and information needed to continue training the model."""

    savepath: Path

    def load_checkpoint(
        self,
        filename_epoch: int,  # epoch number in filename might be different from one of model?
        model: RegressionMultilayerPerceptron,
        optimizer: torch.optim.Optimizer,
    ) -> CheckpointData:
        model_filename = get_model_filename(self.savepath, filename_epoch)
        checkpoint = torch.load(model_filename)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]

        return CheckpointData(model, optimizer, epoch)
