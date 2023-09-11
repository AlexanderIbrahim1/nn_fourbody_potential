"""
This module contains functions used in other directories in the `modelio` package.
"""

from pathlib import Path
from nn_fourbody_potential.models import TrainingParameters


def get_model_filename(savepath: Path, epoch: int) -> Path:
    return Path(savepath, f"nnpes_{epoch:0>5d}.pth")


def write_training_parameters(savepath: Path, params: TrainingParameters, overwrite: bool = False) -> None:
    if savepath.exists() and not overwrite:
        raise FileExistsError(
            "The following file already exists:\n"
            f"{savepath}\n"
            "To overwrite an existing file, pass in 'overwrite=True'\n"
        )

    csv_layers = ", ".join(str(layer) for layer in params.layers)
    repr_transformations = "\n\n".join([repr(trans) for trans in params.transformations])

    with open(savepath, "w") as fout:
        fout.write(f"Seed: {params.seed}\n")
        fout.write(f"Layer sizes: [{csv_layers}]\n")
        fout.write(f"Learning rate: {params.learning_rate:.6f}\n")
        fout.write(f"Weight decay: {params.weight_decay:.6f}\n")
        fout.write(f"Training data set size: {params.training_size}\n")
        fout.write(f"Total epochs: {params.total_epochs}\n")
        fout.write(f"Batch size: {params.batch_size}\n")
        fout.write(f"Batch normalization applied: {params.apply_batch_norm}\n")
        fout.write(f"The transformations used are the following:\n\n{repr_transformations}\n")
        fout.write("\n")
        fout.write(f"Other information:\n{params.other}\n")
