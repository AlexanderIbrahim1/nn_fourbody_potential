"""
This module is for experimenting with the idea of training a neural network using
output energies that are rescaled by an analytic toy exponential decay potential.

The hope is that this rescaling brings all the energies into a narrower range of
values, thus making the training more effective.
"""

from __future__ import annotations

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import copy
from pathlib import Path
from typing import Callable
from typing import Optional
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch  # type: ignore

from dispersion4b.coefficients import c12_parahydrogen_midzuno_kihara
from nn_fourbody_potential.constants import ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_COEFF
from nn_fourbody_potential.constants import ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_EXPON
from nn_fourbody_potential.full_range.extrapolated_potential import ExtrapolatedPotential
from nn_fourbody_potential.modelio import write_training_parameters
from nn_fourbody_potential.modelio.utils import get_model_filename
from nn_fourbody_potential.models import TrainingParameters
from nn_fourbody_potential.models import RegressionMultilayerPerceptron
from nn_fourbody_potential.transformations import SixSideLengthsTransformer
from nn_fourbody_potential import rescaling

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
        batch_size=4096,
        transformations=data_transforms,
        apply_batch_norm=False,
        other=other_info,
    )


def get_toy_decay_potential() -> rescaling.RescalingPotential:
    # constants chosen so that the ratio of the absolute values of the minimum and maximum reduced
    # energies is the lowest possible
    coeff = ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_COEFF / 12.0
    expon = ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_EXPON * 5.02
    disp_coeff = 1.0 * c12_parahydrogen_midzuno_kihara()

    return rescaling.RescalingPotential(coeff, expon, disp_coeff)


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


class InverseEnergyRescaler:
    def __init__(
        self,
        rescaling_potential: rescaling.RescalingPotential,
        res_limits: rescaling.RescalingLimits,
    ) -> None:
        self._rescaling_potential = rescaling_potential
        self._lin_map = rescaling.LinearMap(rescaling.invert_rescaling_limits(res_limits))

    def __call__(self, rescaled_energy: float, *six_pair_distances: float) -> float:
        rescale_value = self._rescaling_potential(*six_pair_distances)
        unscaled_energy = self._lin_map(rescaled_energy) * rescale_value

        return unscaled_energy


# def prepared_data(
#     data_filepath: Path,
#     sidelength_transforms: Sequence[SixSideLengthsTransformer],
#     allow_predicate: Callable[[float, float], bool],
#     rescaling_potential: rescaling.RescalingPotential,
#     rescaled_limits: Optional[rescaling.RescalingLimits] = None,
# ) -> tuple[torch.Tensor, torch.Tensor, rescaling.RescalingLimits]:
#     x_raw, energies_raw = training.prepared_data(data_filepath, sidelength_transforms)
#     sidelengths, _ = training.prepared_data(data_filepath, [])
#
#     x_pruned, energies_pruned = pruned_data(sidelengths, x_raw, energies_raw, allow_predicate)
#     sidelengths_pruned, _ = pruned_data(sidelengths, sidelengths, energies_raw, allow_predicate)
#
#     energies_rescaled = rescaled_energies(rescaling_potential, sidelengths_pruned, energies_pruned)
#
#     if rescaled_limits is None:
#         min_resc_eng = energies_rescaled.min().item()
#         max_resc_eng = energies_rescaled.max().item()
#         res_limits = rescaling.RescalingLimits(min_resc_eng, max_resc_eng, -1.0, 1.0)
#     else:
#         res_limits = copy.deepcopy(rescaled_limits)
#
#     lin_map = rescaling.LinearMap(res_limits)  # noqa
#
#     for i, eng in enumerate(energies_rescaled):
#         energies_rescaled[i] = lin_map(eng)
#
#     return (x_pruned, energies_rescaled, res_limits)


def train_with_rescaling() -> None:
    training_data_filepath = Path("energy_separation", "data", "all_energy_train_filtered.dat")
    testing_data_filepath = Path("energy_separation", "data", "all_energy_test_filtered.dat")
    validation_data_filepath = Path("energy_separation", "data", "all_energy_valid_filtered.dat")
    other_info = "_rescaled_model_all3"

    rescaling_potential = get_toy_decay_potential()

    transforms = model_info.get_data_transforms()
    params = get_training_parameters(training_data_filepath, transforms, other_info)

    side_length_groups_train, energies_train = training.load_fourbody_training_data(training_data_filepath)
    side_length_groups_test, energies_test = training.load_fourbody_training_data(testing_data_filepath)
    side_length_groups_valid, energies_valid = training.load_fourbody_training_data(validation_data_filepath)

    x_train, y_train, res_limits = rescaling.prepare_rescaled_data(
        side_length_groups_train, energies_train, transforms, rescaling_potential
    )
    print(res_limits)

    x_test, y_test = rescaling.prepare_rescaled_data_with_rescaling_limits(
        side_length_groups_test, energies_test, transforms, rescaling_potential, res_limits
    )

    x_valid, y_valid = rescaling.prepare_rescaled_data_with_rescaling_limits(
        side_length_groups_valid, energies_valid, transforms, rescaling_potential, res_limits
    )

    model = RegressionMultilayerPerceptron(training.N_FEATURES, training.N_OUTPUTS, params.layers)

    # moving everything to the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    x_valid = x_valid.to(device)
    y_valid = y_valid.to(device)
    model = model.to(device)

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


class RescalingEnergyModel:
    def __init__(
        self,
        max_n_samples: int,
        rescaled_model: RegressionMultilayerPerceptron,
        inverse_rescaler: InverseEnergyRescaler,
    ) -> None:
        self._check_max_n_samples(max_n_samples)

        self._max_n_samples = max_n_samples
        self._rescaled_model = rescaled_model
        self._inverse_rescaler = inverse_rescaler

    def __call__(self, samples: torch.Tensor, sidelength_groups: torch.Tensor) -> torch.Tensor:
        n_samples = len(samples)

        if n_samples == 0:
            return torch.Tensor([])

        self._rescaled_model.eval()
        with torch.no_grad():
            rescaled_energies = self._rescaled_model.forward(samples)

        for i, (eng, sidelengths) in enumerate(zip(rescaled_energies, sidelength_groups)):
            tup_sidelengths = tuple([s.item() for s in sidelengths])
            rescaled_energies[i] = self._inverse_rescaler(eng.item(), *tup_sidelengths)

        return rescaled_energies

    def eval(self) -> None:
        pass

    def _check_max_n_samples(self, max_n_samples: int) -> None:
        if max_n_samples <= 0:
            raise ValueError("The buffer size for the calculations must be a positive number.")


if __name__ == "__main__":
    train_with_rescaling()
#     res_limits = rescaling.RescalingLimits(
#         from_left=-3.100146532058716, from_right=8.271796226501465, to_left=-1.0, to_right=1.0
#     )
#     rescaling_potential = get_toy_decay_potential()
#     inv_eng_rescaler = InverseEnergyRescaler(rescaling_potential, res_limits)
#
#     model_filename = Path(
#         "/home/a68ibrah/research/four_body_interactions/nn_fourbody_potential/app/models/nnpes_rescaled_model_all2_layers64_128_128_64_lr_0.000500_datasize_15101/models/nnpes_00999.pth"
#     )
#     checkpoint = torch.load(model_filename)
#     model = RegressionMultilayerPerceptron(training.N_FEATURES, training.N_OUTPUTS, [64, 128, 128, 64])
#     model.load_state_dict(checkpoint["model_state_dict"])
#
#     energy_model = RescalingEnergyModel(1024, model, inv_eng_rescaler)
#
#     transforms = model_info.get_data_transforms()
#     extrapolated_potential = ExtrapolatedPotential(energy_model, transforms, pass_in_sidelengths_to_network=True)
#
#     sidelengths = np.linspace(1.9, 5.0, 256)
#     sidelength_groups = np.array([(s, s, s, s, s, s) for s in sidelengths]).reshape(-1, 6).astype(np.float32)
#     output_energies = extrapolated_potential.evaluate_batch(sidelength_groups)
#
#     _, ax = plt.subplots()
#     ax.plot(sidelengths, output_energies)
#     plt.show()
