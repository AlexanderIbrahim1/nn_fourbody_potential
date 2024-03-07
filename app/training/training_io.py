"""
This module contains functions for managing the files and save paths of models
during training.
"""

from pathlib import Path

from nn_fourbody_potential.models import TrainingParameters


def number_of_lines(file: Path) -> int:
    with open(file, "r") as fin:
        return sum([1 for _ in fin])


def model_directory_name(layers: list[int], learning_rate: float, training_size: int, other: str = "") -> Path:
    """The name of the directory where this set of models will be saved."""
    layer_part = "layers" + "_".join([str(layer) for layer in layers])
    lr_part = f"lr_{learning_rate:.6f}"
    training_size_part = f"datasize_{training_size}"

    return f"{other}_{layer_part}_{lr_part}_{training_size_part}"


def get_path_to_model(params: TrainingParameters, base_dirpath: Path) -> Path:
    base_models_path = base_dirpath / "models"
    specific_model_dir = model_directory_name(
        params.layers, params.learning_rate, params.training_size, other=f"nnpes{params.other}"
    )

    return base_models_path / specific_model_dir


def get_saved_models_dirpath(params: TrainingParameters, base_dirpath: Path) -> Path:
    modelpath = get_path_to_model(params, base_dirpath)
    return modelpath / "models"


def get_model_filename(savepath: Path, epoch: int) -> Path:
    return Path(savepath, f"nnpes_{epoch:0>5d}.pth")


def get_training_parameters_filepath(params: TrainingParameters, base_dirpath: Path) -> Path:
    modelpath = get_path_to_model(params, base_dirpath)
    return modelpath / "training_parameters.dat"


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
