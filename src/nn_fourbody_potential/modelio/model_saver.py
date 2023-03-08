
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


def write_training_parameters(savepath: Path, params: TrainingParameters, overwrite: bool = False) -> None:
    if savepath.exists() and not overwrite:
        raise FileExistsError(
            "The following file already exists:\n"
            f"{savepath}\n"
            "To overwrite an existing file, pass in 'overwrite=True'\n"
        )
    
    csv_layers = ", ".join(str(layer) for layer in params.layers)
    repr_transformations = "\n\n".join([repr(trans) for trans in params.transformations])
    
    with open(savepath, 'w') as fout:
        fout.write(f"Seed: {params.seed}\n")
        fout.write(f"Layer sizes: [{csv_layers}]\n")
        fout.write(f"Learning rate: {params.learning_rate:.6f}\n")
        fout.write(f"Weight decay: {params.weight_decay:.6f}\n")
        fout.write(f"Training data set size: {params.training_size}\n")
        fout.write(f"Total epochs: {params.total_epochs}\n")
        fout.write(f"Batch size: {params.batch_size}\n")
        fout.write(f"The transformations used are the following:\n\n{repr_transformations}\n")
        fout.write('\n')
        fout.write(f"Other information:\n{params.other}\n")