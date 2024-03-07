from dataclasses import dataclass
from pathlib import Path

import torch

from nn_fourbody_potential.models import RegressionMultilayerPerceptron
from nn_fourbody_potential.models import TrainingParameters


@dataclass(frozen=True)
class ModelSaver:
    """Save the .pth files based on their epoch number"""

    savepath: Path

    def save_model(self, model: RegressionMultilayerPerceptron, *, epoch: int) -> None:
        assert epoch >= 0

        if not self.savepath.exists():
            self.savepath.mkdir()

        model_filename = self.get_model_filename(epoch)
        torch.save(model.state_dict(), model_filename)

    def get_model_filename(self, epoch: int) -> Path:
        return Path(self.savepath, f"nnpes_{epoch:0>5d}.pth")
