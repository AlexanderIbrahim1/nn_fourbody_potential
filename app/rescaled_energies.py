"""
This module is for experimenting with the idea of training a neural network using
output energies that are rescaled by an analytic toy exponential decay potential.

The hope is that this rescaling brings all the energies into a narrower range of
values, thus making the training more effective.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Optional
from typing import Sequence

import numpy as np
import torch  # type: ignore

from dispersion4b.coefficients import c12_parahydrogen_midzuno_kihara
from nn_fourbody_potential.constants import ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_COEFF
from nn_fourbody_potential.constants import ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_EXPON
from nn_fourbody_potential.modelio import write_training_parameters
from nn_fourbody_potential.modelio.utils import get_model_filename
from nn_fourbody_potential.models import TrainingParameters
from nn_fourbody_potential.models import RegressionMultilayerPerceptron
from nn_fourbody_potential.rescaling_potential import RescalingPotential
from nn_fourbody_potential.transformations import SixSideLengthsTransformer

import model_info
import training


def get_training_parameters(
    data_filepath: Path,
    data_transforms: list[SixSideLengthsTransformer],
    other_info: str,
) -> TrainingParameters:
    return TrainingParameters(
        seed=0,
        layers=[64, 128, 128, 64],
        learning_rate=2.0e-4,
        weight_decay=1.0e-4,
        training_size=model_info.number_of_lines(data_filepath),
        total_epochs=1000,
        batch_size=2000,
        transformations=data_transforms,
        apply_batch_norm=False,
        other=other_info,
    )


def get_toy_decay_potential() -> RescalingPotential:
    coeff = ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_COEFF / 3.0
    expon = ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_EXPON * 6.0
    disp_coeff = 20.0 * c12_parahydrogen_midzuno_kihara()

    return RescalingPotential(coeff, expon, disp_coeff)


def rescaled_energies(
    potential: RescalingPotential, side_length_groups: torch.Tensor, energies: torch.Tensor
) -> torch.Tensor:
    rescaled = copy.deepcopy(energies)

    for i, side_lengths in enumerate(side_length_groups):
        rescale_value = potential(*(s.item() for s in side_lengths))
        rescaled[i] /= rescale_value

    return rescaled


def pruned_data(
    sidelength_groups: torch.Tensor,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    allow_predicate: Callable[[float, float], bool],
) -> tuple[torch.Tensor, torch.Tensor]:
    filtered = [
        i
        for (i, (energy, sidelengths)) in enumerate(zip(y_data, sidelength_groups))
        if allow_predicate(energy, sidelengths.mean().item())
    ]

    x_data_filtered = x_data[filtered]
    y_data_filtered = y_data[filtered]

    return x_data_filtered, y_data_filtered


def linear_map(from_left: float, from_right: float, to_left: float, to_right: float) -> Callable[[float], float]:
    slope = (to_right - to_left) / (from_right - from_left)

    return lambda x: (x - from_left) * slope + to_left


@dataclass
class RescaledLimits:
    lower: float
    upper: float


def prepared_data(
    data_filepath: Path,
    sidelength_transforms: Sequence[SixSideLengthsTransformer],
    allow_predicate: Callable[[float, float], bool],
    rescaling_potential: RescalingPotential,
    rescaled_limits: Optional[RescaledLimits] = None,
) -> tuple[torch.Tensor, torch.Tensor, RescaledLimits]:
    x_raw, energies_raw = training.prepared_data(data_filepath, sidelength_transforms)
    sidelengths, _ = training.prepared_data(data_filepath, [])

    x_pruned, energies_pruned = pruned_data(sidelengths, x_raw, energies_raw, allow_predicate)
    sidelengths_pruned, _ = pruned_data(sidelengths, sidelengths, energies_raw, allow_predicate)

    energies_rescaled = rescaled_energies(rescaling_potential, sidelengths_pruned, energies_pruned)

    if rescaled_limits is None:
        min_resc_eng = energies_rescaled.min().item()
        max_resc_eng = energies_rescaled.max().item()
    else:
        min_resc_eng = rescaled_limits.lower
        max_resc_eng = rescaled_limits.upper

    ret_limits = RescaledLimits(min_resc_eng, max_resc_eng)
    lin_map = linear_map(min_resc_eng, max_resc_eng, -1.0, 1.0)

    for i, eng in enumerate(energies_rescaled):
        energies_rescaled[i] = lin_map(eng)

    return (x_pruned, energies_rescaled, ret_limits)


# create function that gets prepared data, but also:
# - applies a filter to prune small energies
# - applies the rescaling potential
# - applies a normalization (and returns the normalization limits)
#
# the rescaling potential, obviously, rescales the energies
# - originally, there are energies from ~200 wvn to 1.0e-3 wvn (5 OOM)
# - hopefully, the rescaling will put them all within 1 or 2 OOM
# - the relative weight between the exponential and dispersion parts can be used to emphasize the relative importance of small and large energies
#
# NOTE: the long range pruning should take into account the mixing between the NN and the dispersion potential


def train_with_rescaling() -> None:
    training_data_filepath = Path("energy_separation", "data", "all_energy_train.dat")
    testing_data_filepath = Path("energy_separation", "data", "all_energy_test.dat")
    validation_data_filepath = Path("energy_separation", "data", "all_energy_valid.dat")
    other_info = "_rescaled_model_all"

    allow_predicate = lambda energy, avg_dist: abs(energy) >= 1.0e-3 and avg_dist <= 4.5
    rescaling_potential = get_toy_decay_potential()

    transforms = model_info.get_data_transforms()
    params = get_training_parameters(training_data_filepath, transforms, other_info)

    x_train, y_train, ret_limits = prepared_data(
        training_data_filepath, transforms, allow_predicate, rescaling_potential
    )
    x_test, y_test, _ = prepared_data(
        testing_data_filepath, transforms, allow_predicate, rescaling_potential, ret_limits
    )
    x_valid, y_valid, _ = prepared_data(
        validation_data_filepath, transforms, allow_predicate, rescaling_potential, ret_limits
    )

    print(ret_limits)

    model = RegressionMultilayerPerceptron(training.N_FEATURES, training.N_OUTPUTS, params.layers)

    modelpath = model_info.get_path_to_model(params)
    if not modelpath.exists():
        modelpath.mkdir()

    if not (params_filepath := model_info.get_training_parameters_filepath(params)).exists():
        write_training_parameters(params_filepath, params, overwrite=False)

    saved_models_dirpath = model_info.get_saved_models_dirpath(params)

    training.train_model(
        x_train,
        y_train,
        x_valid,
        y_valid,
        params,
        model,
        modelpath,
        save_every=50,
        # continue_training_from_epoch=5200,
    )

    last_model_filename = get_model_filename(saved_models_dirpath, params.total_epochs - 1)
    test_loss = training.test_model(x_test, y_test, model, last_model_filename)
    print(f"test loss mse = {test_loss}")
    print(f"test loss rmse = {np.sqrt(test_loss)}")


if __name__ == "__main__":
    train_with_rescaling()

#
#
# def train_with_fast_decay_data() -> None:
#     training_data_filepath = model_info.get_training_data_filepath()
#     hcp_data_filepath = model_info.get_hcp_data_filepath()
#     validation_data_filepath = model_info.get_validation_data_filepath()
#     testing_data_filepath = model_info.get_testing_data_filepath()
#
#     fastdecay_training_data_filepath = model_info.get_fastdecay_training_data_filepath()
#     fastdecay_testing_data_filepath = model_info.get_fastdecay_testing_data_filepath()
#     fastdecay_validation_data_filepath = model_info.get_fastdecay_validation_data_filepath()
#     veryfastdecay_training_data_filepath = model_info.get_veryfastdecay_training_data_filepath()
#     veryfastdecay_testing_data_filepath = model_info.get_veryfastdecay_testing_data_filepath()
#     veryfastdecay_validation_data_filepath = model_info.get_veryfastdecay_validation_data_filepath()
#
#     transforms = model_info.get_data_transforms()
#     params = model_info.get_training_parameters(training_data_filepath, transforms)
#
#     x_train_hcp, y_train_hcp = training.prepared_data(hcp_data_filepath, transforms)
#     x_train_gen, y_train_gen = training.prepared_data(training_data_filepath, transforms)
#     x_valid, y_valid = training.prepared_data(validation_data_filepath, transforms)
#     x_test, y_test = training.prepared_data(testing_data_filepath, transforms)
#     x_fastdecay_test, y_fastdecay_test = training.prepared_data(fastdecay_testing_data_filepath, transforms)
#     x_fastdecay_train, y_fastdecay_train = training.prepared_data(fastdecay_training_data_filepath, transforms)
#     x_fastdecay_valid, y_fastdecay_valid = training.prepared_data(fastdecay_validation_data_filepath, transforms)
#     x_veryfastdecay_test, y_veryfastdecay_test = training.prepared_data(veryfastdecay_testing_data_filepath, transforms)
#     x_veryfastdecay_train, y_veryfastdecay_train = training.prepared_data(
#         veryfastdecay_training_data_filepath, transforms
#     )
#     x_veryfastdecay_valid, y_veryfastdecay_valid = training.prepared_data(
#         veryfastdecay_validation_data_filepath, transforms
#     )
#
#     # hcp_mask = torch.Tensor(
#     #     [
#     #         prune.sidelengths_filter(sidelens, 4.2) and prune.energy_filter(energy, 1.0e-3)
#     #         for (sidelens, energy) in zip(x_train_hcp, y_train_hcp)
#     #     ]
#     # )
#     # x_train_hcp = x_train_hcp[torch.nonzero(hcp_mask).reshape(-1)]
#     # y_train_hcp = y_train_hcp[torch.nonzero(hcp_mask).reshape(-1)]
#
#     x_train = torch.concatenate((x_train_gen, x_train_hcp, x_fastdecay_train, x_veryfastdecay_train))
#     y_train = torch.concatenate((y_train_gen, y_train_hcp, y_fastdecay_train, y_veryfastdecay_train))
#     x_valid = torch.concatenate((x_valid, x_fastdecay_valid, x_veryfastdecay_valid))
#     y_valid = torch.concatenate((y_valid, y_fastdecay_valid, y_veryfastdecay_valid))
#     x_test = torch.concatenate((x_test, x_fastdecay_test, x_veryfastdecay_test))
#     y_test = torch.concatenate((y_test, y_fastdecay_test, y_veryfastdecay_test))
#
#     model = RegressionMultilayerPerceptron(training.N_FEATURES, training.N_OUTPUTS, params.layers)
#
#     modelpath = model_info.get_path_to_model(params)
#     if not modelpath.exists():
#         modelpath.mkdir()
#
#     training_parameters_filepath = model_info.get_training_parameters_filepath(params)
#
#     saved_models_dirpath = model_info.get_saved_models_dirpath(params)
#     model_saver = ModelSaver(saved_models_dirpath)
#
#     write_training_parameters(training_parameters_filepath, params, overwrite=False)
#     training.train_model(
#         x_train,
#         y_train,
#         x_valid,
#         y_valid,
#         params,
#         model,
#         modelpath,
#         model_saver,
#         save_every=20,
#     )
#
#     last_model_filename = model_saver.get_model_filename(params.total_epochs - 1)
#     test_loss = training.test_model(x_test, y_test, model, last_model_filename)
#     print(f"test loss mse = {test_loss}")
#     print(f"test loss rmse = {np.sqrt(test_loss)}")
#
#
# if __name__ == "__main__":
#     train_with_fast_decay_data()
#
