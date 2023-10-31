"""
This script compares the testing errors of different models.
"""

import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from nn_fourbody_potential import rescaling
from nn_fourbody_potential.energy_scale.energy_scale_ensemble_model import EnergyScaleEnsembleModel
from nn_fourbody_potential.energy_scale.energy_scale import EnergyScaleAssigner
from nn_fourbody_potential.full_range.extrapolated_potential import ExtrapolatedPotential
from nn_fourbody_potential.models import RegressionMultilayerPerceptron

from dispersion4b.coefficients import c12_parahydrogen_midzuno_kihara
from nn_fourbody_potential.constants import ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_COEFF
from nn_fourbody_potential.constants import ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_EXPON

import model_info
import training


def get_ensemble_model_filepath(which: str) -> Path:
    datasizes = {"all": "15101", "low": "7941", "mid": "3879", "high": "4393"}

    return Path(
        "energy_separation",
        "models",
        f"nnpes_{which}_energies2_layers64_128_128_64_lr_0.000100_datasize_{datasizes[which]}",
        "models",
        "nnpes_09950.pth",
    )


def get_model(model_filepath: Path, layer_sizes: list[int]) -> RegressionMultilayerPerceptron:
    n_features = 6
    n_outputs = 1
    checkpoint_dict = torch.load(model_filepath)
    model = RegressionMultilayerPerceptron(n_features, n_outputs, layer_sizes)
    model.load_state_dict(checkpoint_dict["model_state_dict"])

    return model


def get_energy_scale_ensemble_potential() -> ExtrapolatedPotential:
    assigner = EnergyScaleAssigner(
        low_medium_centre=1.0,
        low_medium_width=0.05,
        medium_high_centre=10.0,
        medium_high_width=0.5,
    )

    max_n_samples = 4096
    layer_sizes = [64, 128, 128, 64]

    all_model_filepath = get_ensemble_model_filepath("all")
    low_model_filepath = get_ensemble_model_filepath("low")
    mid_model_filepath = get_ensemble_model_filepath("mid")
    high_model_filepath = get_ensemble_model_filepath("high")

    all_model = get_model(all_model_filepath, layer_sizes)
    low_model = get_model(low_model_filepath, layer_sizes)
    mid_model = get_model(mid_model_filepath, layer_sizes)
    high_model = get_model(high_model_filepath, layer_sizes)

    ensemble_model = EnergyScaleEnsembleModel(max_n_samples, all_model, low_model, mid_model, high_model, assigner)

    return ExtrapolatedPotential(ensemble_model, model_info.get_data_transforms())


def get_reverse_rescaler() -> rescaling.ReverseEnergyRescaler:
    res_limits = rescaling.RescalingLimits(
        from_left=-3.2540090084075928, from_right=8.625899314880371, to_left=-1.0, to_right=1.0
    )
    rev_res_limits = rescaling.invert_rescaling_limits(res_limits)

    rescaling_function = rescaling.RescalingPotential(
        coeff=ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_COEFF / 12.0,
        expon=ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_EXPON * 5.02,
        disp_coeff=1.0 * c12_parahydrogen_midzuno_kihara(),
    )

    return rescaling.ReverseEnergyRescaler(rescaling_function, rev_res_limits)


def get_rescaling_potential() -> ExtrapolatedPotential:
    # model_category = "nnpes_rescaling_model4_layers64_128_128_64_lr_0.000200_datasize_12621"
    # model_filepath = Path("models", model_category, "models", "nnpes_19999.pth")
    # model = get_model(model_filepath, [64, 128, 128, 64])

    # model_category = "nnpes_rescaling_model_filtered10_layers64_128_128_64_lr_0.000200_datasize_13610"
    model_category = "nnpes_rescaling_model_filtered11_layers32_64_64_32_lr_0.000200_datasize_13610"
    model_filepath = Path("models", model_category, "models", "nnpes_09999.pth")
    model = get_model(model_filepath, [32, 64, 64, 32])

    rev_rescaler = get_reverse_rescaler()

    energy_model = rescaling.RescalingEnergyModel(model, rev_rescaler)

    return ExtrapolatedPotential(
        energy_model, model_info.get_data_transforms_flattening(), pass_in_sidelengths_to_network=True
    )


def get_rms_error(x0_data: Sequence[float], x1_data: Sequence[float]) -> float:
    assert len(x0_data) == len(x1_data)

    total_sum_of_squares = sum([(x0 - x1) ** 2 for (x0, x1) in zip(x0_data, x1_data)])
    average_of_squares = total_sum_of_squares / len(x0_data)

    return math.sqrt(average_of_squares)


def main() -> None:
    ensemble_potential = get_energy_scale_ensemble_potential()
    rescaling_potential = get_rescaling_potential()

    # NOTE: this isn't a fair comparison, because I modified the splits, and so the ensemble model would have
    # been trained on some of the testing data of the new model. But this is just an informal look at the
    # two models, so I don't care too much.
    testing_data_filepath = Path("energy_separation", "data_splitting", "split_data", "test.dat")
    side_length_groups_test, energies_test = training.load_fourbody_training_data(testing_data_filepath)

    output_energies_ensemble = ensemble_potential.evaluate_batch(side_length_groups_test)
    output_energies_rescaling = rescaling_potential.evaluate_batch(side_length_groups_test)

    # savedata = np.vstack((energies_test, output_energies_rescaling)).T
    # save_filepath = Path(".", "test_and_rescaling_energies.dat")
    # np.savetxt(save_filepath, savedata)
    # exit()

    print(get_rms_error(output_energies_ensemble, energies_test))
    print(get_rms_error(output_energies_rescaling, energies_test))

    _, ax = plt.subplots()
    ax.plot([-5, 180], [-5, 180])
    ax.plot(energies_test, output_energies_ensemble, "C1s", lw=0, ms=3, label="ensemble")
    ax.plot(energies_test, output_energies_rescaling, "C2o", lw=0, ms=3, label="rescaling")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
