from dataclasses import dataclass
from pathlib import Path

import torch

from nn_fourbody_potential.models import RegressionMultilayerPerceptron
from nn_fourbody_potential.modelio.utils import get_model_filename


@dataclass(frozen=True)
class CheckpointSaver:
    """
    Save the states of the model and the optimizer, and other relevant information
    needed to continue training at a later time.
    """

    savepath: Path

    def save_checkpoint(
        self, *, model: RegressionMultilayerPerceptron, optimizer: torch.optim.Optimizer, epoch: int, loss: float
    ) -> None:
        assert epoch >= 0
        assert loss >= 0.0

        if not self.savepath.exists():
            self.savepath.mkdir()

        model_filename = get_model_filename(self.savepath, epoch)

        checkpoint_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
        }

        torch.save(checkpoint_dict, model_filename)
