import dataclasses
from pathlib import Path
from typing import Any
from typing import Union

import torch
import numpy as np

from nn_fourbody_potential.models import RegressionMultilayerPerceptron
from nn_fourbody_potential.modelio import get_model_filename


@dataclasses.dataclass
class TrainingStateData:
    model: RegressionMultilayerPerceptron
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    epoch: int


def create_training_state_dict(
    model: RegressionMultilayerPerceptron,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
) -> dict[str, Any]:
    if epoch < 0:
        raise RuntimeError(f"A training state can only be created for non-negative epochs. Found: {epoch}")

    training_state_dict = {
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
    }

    return training_state_dict


def save_training_state_dict(savepath: Union[Path, str], training_state_dict: dict[str, Any]) -> None:
    savepath_ = Path(savepath)
    if not savepath_.exists():
        savepath_.mkdir()

    model_filename = get_model_filename(savepath_, training_state_dict["epoch"])
    torch.save(training_state_dict, model_filename)


def load_training_state_dict(savepath: Union[Path, str], epoch: int) -> dict[str, Any]:
    model_filename = get_model_filename(Path(savepath), epoch)
    training_state_dict = torch.load(model_filename)

    state_dict_epoch = training_state_dict["epoch"]
    if epoch != state_dict_epoch:
        raise RuntimeError(
            "Epoch passed in does not match epoch of the training state.\n"
            f"Epoch passed in: {epoch}\n"
            f"Epoch in training_state_dict: {state_dict_epoch}"
        )

    return training_state_dict


def apply_training_state_dict(
    training_state_dict: dict[str, Any],
    model: RegressionMultilayerPerceptron,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> TrainingStateData:
    torch.set_rng_state(training_state_dict["torch_rng_state"])
    np.random.set_state(training_state_dict["numpy_rng_state"])

    model.load_state_dict(training_state_dict["model_state_dict"])
    optimizer.load_state_dict(training_state_dict["optimizer_state_dict"])
    scheduler.load_state_dict(training_state_dict["scheduler_state_dict"])
    epoch = training_state_dict["epoch"]

    return TrainingStateData(model, optimizer, scheduler, epoch)
