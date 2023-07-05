"""
Play around with the energy extrapolation code.
"""
from pathlib import Path

import torch

import matplotlib.pyplot as plt
import numpy as np

from cartesian import Cartesian3D
from cartesian.operations import relative_pair_distances

from nn_fourbody_potential.full_range import ExtrapolatedPotential
from nn_fourbody_potential.full_range import InteractionRange
from nn_fourbody_potential.models import RegressionMultilayerPerceptron
from nn_fourbody_potential.sidelength_distributions.sidelength_types import SixSideLengths
from nn_fourbody_potential.transformations.transformers import MinimumPermutationTransformer
from nn_fourbody_potential.transformations.transformers import ReciprocalTransformer
from nn_fourbody_potential.transformations.transformers import SixSideLengthsTransformer
from nn_fourbody_potential.transformations.transformers import StandardizeTransformer
from nn_fourbody_potential.transformations import transform_sidelengths_data


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


def get_model() -> RegressionMultilayerPerceptron:
    # create the PyTorch model; the following parameters (input features, outputs, and the layer sizes)
    # are specific to the model that was trained;
    # so far, the weights have not been initialized
    n_features = 6
    n_outputs = 1
    layer_sizes = [64, 128, 128, 64]
    # layer_sizes = [128, 256, 256, 128]
    # layer_sizes = [64, 128, 256, 512, 256, 128, 64]
    model = RegressionMultilayerPerceptron(n_features, n_outputs, layer_sizes)

    # the path to the specific .pth file
    # 2999 corresponds to the very last batch
    modelfile = Path(
        "models",
        "nnpes_initial_layers64_128_128_64_lr_0.000200_datasize_8901",
        # "nnpes_small128_withfast_nomask_layers128_256_256_128_lr_0.000200_datasize_5000",
        # "nnpes_big_withfast_nomask_layers64_128_256_512_256_128_64_lr_0.000100_datasize_5000",
        # "nnpes_small64_withfast_layers64_128_128_64_lr_0.000200_datasize_5000",
        "models",
        "nnpes_02999.pth",
    )

    # fill the weights of the model
    model.load_state_dict(torch.load(modelfile))

    return model


def feature_transformers() -> list[SixSideLengthsTransformer]:
    min_sidelen = 2.2
    max_sidelen = 4.5

    return [
        ReciprocalTransformer(),
        StandardizeTransformer((1.0 / max_sidelen, 1.0 / min_sidelen), (0.0, 1.0)),
        MinimumPermutationTransformer(),
    ]


def main() -> None:
    transformers = feature_transformers()
    model = get_model()
    potential = ExtrapolatedPotential(model, transformers)

    lattice_constants = np.linspace(1.5, 5.0, 256)
    samples = [get_sample(lat_const) for lat_const in lattice_constants]
    energies = potential.evaluate_batch(samples)

    transformed_samples = transform_sidelengths_data(samples, transformers)
    transformed_samples = torch.from_numpy(transformed_samples.astype(np.float32))

    with torch.no_grad():
        output_data = model(transformed_samples)
        output_energies = output_data.detach().cpu().numpy()  # NOTE: is .cpu() making too many assumptions?

    fig, ax = plt.subplots()

    ax.set_xlim(3.0, 5.0)
    ax.set_ylim(-0.06, 0.08)

    ax.set_title(r"Four-Body Interaction Energy of a Tetrahedron", fontsize=14)
    ax.set_ylabel(r"$ V_4 \ / \ \mathrm{cm}^{-1} $", fontsize=16)
    ax.set_xlabel(r"lattice constant / $ \AA $", fontsize=16)

    ax.plot(lattice_constants, energies, label="Extrapolated Potential")
    ax.plot(lattice_constants, output_energies, label="Neural Network, no extrapolation")

    ax.legend(fontsize=12)

    fig.tight_layout()
    # plt.show()
    plt.savefig("figures/potential_extrapolation_zoom_in.pdf", dpi=400)


if __name__ == "__main__":
    main()
