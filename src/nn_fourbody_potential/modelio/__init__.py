"""
The 'modelio' subpackage contains code for managing the saving of models and other
data related to the models.
"""
from pathlib import Path

from nn_fourbody_potential.modelio.checkpoint_loader import CheckpointLoader
from nn_fourbody_potential.modelio.checkpoint_saver import CheckpointSaver
from nn_fourbody_potential.modelio.error_writer import ErrorWriter
from nn_fourbody_potential.modelio.model_saver import ModelSaver
from nn_fourbody_potential.modelio.utils import write_training_parameters


def model_directory_name(layers: list[int], learning_rate: float, training_size: int, other: str = "") -> Path:
    """The name of the directory where this set of models will be saved."""
    layer_part = "layers" + "_".join([str(layer) for layer in layers])
    lr_part = f"lr_{learning_rate:.6f}"
    training_size_part = f"datasize_{training_size}"

    return f"{other}_{layer_part}_{lr_part}_{training_size_part}"
