"""
This module contains code for plotting the piecewise ensemble potential, which uses
a combination of three neural networks (for low, medium, and high energies) to create
a single potential.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from cartesian import Cartesian3D
from cartesian.operations import relative_pair_distances

from nn_fourbody_potential.energy_scale.energy_scale_ensemble_model import EnergyScaleEnsembleModel
from nn_fourbody_potential.energy_scale.energy_scale import EnergyScaleAssigner
from nn_fourbody_potential.full_range.extrapolated_potential import ExtrapolatedPotential
from nn_fourbody_potential.models import RegressionMultilayerPerceptron
from nn_fourbody_potential.sidelength_distributions.sidelength_types import SixSideLengths
from nn_fourbody_potential.transformations.applications import transform_sidelengths_data
from nn_fourbody_potential.transformations.transformers import MinimumPermutationTransformer
from nn_fourbody_potential.transformations.transformers import ReciprocalTransformer
from nn_fourbody_potential.transformations.transformers import SixSideLengthsTransformer
from nn_fourbody_potential.transformations.transformers import StandardizeTransformer


def create_energy_scale_assigner() -> EnergyScaleAssigner:
    return EnergyScaleAssigner(
        low_medium_centre=1.0,
        low_medium_width=0.05,
        medium_high_centre=10.0,
        medium_high_width=0.5,
    )


def get_model(model_filepath: Path, layer_sizes: list[int]) -> RegressionMultilayerPerceptron:
    n_features = 6
    n_outputs = 1
    model = RegressionMultilayerPerceptron(n_features, n_outputs, layer_sizes)
    model.load_state_dict(torch.load(model_filepath))

    return model


def feature_transformers() -> list[SixSideLengthsTransformer]:
    min_sidelen = 2.2
    max_sidelen = 4.5

    return [
        ReciprocalTransformer(),
        StandardizeTransformer((1.0 / max_sidelen, 1.0 / min_sidelen), (0.0, 1.0)),
        MinimumPermutationTransformer(),
    ]


def get_energy_scale_ensemble_model() -> EnergyScaleEnsembleModel:
    max_n_samples = 1024
    energy_scale_assigner = create_energy_scale_assigner()

    layer_sizes = [64, 128, 128, 64]
    all_energy_model_filepath = Path(
        "energy_separation",
        "models",
        "nnpes_all_energies_layers64_128_128_64_lr_0.000200_datasize_15101",
        "models",
        "nnpes_02999.pth",
    )
    low_energy_model_filepath = Path(
        "energy_separation",
        "models",
        "nnpes_low_energies_layers64_128_128_64_lr_0.000200_datasize_7941",
        "models",
        "nnpes_02999.pth",
    )
    mid_energy_model_filepath = Path(
        "energy_separation",
        "models",
        "nnpes_mid_energies_layers64_128_128_64_lr_0.000200_datasize_3879",
        "models",
        "nnpes_02999.pth",
    )
    high_energy_model_filepath = Path(
        "energy_separation",
        "models",
        "nnpes_high_energies_layers64_128_128_64_lr_0.000200_datasize_4393",
        "models",
        "nnpes_02999.pth",
    )

    all_energy_model = get_model(all_energy_model_filepath, layer_sizes)
    low_energy_model = get_model(low_energy_model_filepath, layer_sizes)
    mid_energy_model = get_model(mid_energy_model_filepath, layer_sizes)
    high_energy_model = get_model(high_energy_model_filepath, layer_sizes)

    transformers = feature_transformers()

    return EnergyScaleEnsembleModel(
        max_n_samples,
        all_energy_model,
        low_energy_model,
        mid_energy_model,
        high_energy_model,
        energy_scale_assigner,
        transformers,
    )


def get_sample(lattice_constant: float) -> SixSideLengths:
    # begin with the four points in 3D Cartesian space
    # the values chosen correspond to (almost) a tetrahedron with a side length of 2.2 Angstroms
    assert lattice_constant > 0.0

    points = [
        lattice_constant * Cartesian3D(0.000, 0.000, 0.000),
        lattice_constant * Cartesian3D(1.000, 0.000, 0.000),
        lattice_constant * Cartesian3D(0.500, 0.866, 0.000),
        lattice_constant * Cartesian3D(0.500, 0.289, 0.816),
    ]

    # get the six relative pair distances (r01, r02, r03, r12, r13, r23)
    pair_distances = relative_pair_distances(points)

    return tuple(pair_distances)


def main() -> None:
    ensemble_model = get_energy_scale_ensemble_model()
    extrapolated_potential = ExtrapolatedPotential(ensemble_model, feature_transformers())

    lattice_constants = np.linspace(1.5, 5.0, 256)
    # lattice_constants = np.array([])
    # samples = np.array([get_sample(lat_const) for lat_const in lattice_constants]).reshape(-1, 6).astype(np.float32)
    # samples = torch.from_numpy(samples)
    samples = np.array([get_sample(lat_const) for lat_const in lattice_constants]).reshape(-1, 6).astype(np.float32)

    output_energies = extrapolated_potential.evaluate_batch(samples)

    _, ax = plt.subplots()
    ax.plot(lattice_constants, output_energies)
    plt.show()


if __name__ == "__main__":
    main()
