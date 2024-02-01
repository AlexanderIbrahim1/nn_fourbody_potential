"""
This module contains code to plot the potential energy surfaces (PES) along paths in
coordinate space. The PESs can be a NN PES, or the analytic PES.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated
from typing import Optional
from typing import Sequence

import torch
import numpy as np
import matplotlib.pyplot as plt

from cartesian import Cartesian3D

from hydro4b_coords.geometries import MAP_GEOMETRY_TAG_TO_FUNCTION

from nn_fourbody_potential.fourbody_potential import create_fourbody_analytic_potential
from nn_fourbody_potential.models import RegressionMultilayerPerceptron
from nn_fourbody_potential.sidelength_distributions.generate import get_sidelengths
from nn_fourbody_potential.transformations import SixSideLengthsTransformer
from nn_fourbody_potential.transformations import transform_sidelengths_data

import model_info


FourCartesianPoints = Annotated[Sequence[Cartesian3D], 4]


def get_analytic_energies(
    groups_of_points: Sequence[FourCartesianPoints],
) -> np.ndarray[float]:
    """Get the potential energies from the analytic PES specific to this project."""
    potential = create_fourbody_analytic_potential()
    return np.array([potential(points) for points in groups_of_points])


def get_nn_model_energies(
    groups_of_points: Sequence[FourCartesianPoints],
    model: RegressionMultilayerPerceptron,
    modelfile: Path,
    data_transforms: Optional[list[SixSideLengthsTransformer]] = None,
) -> np.ndarray[float]:
    """Get the potential energies from a trained neural network model."""

    model.load_state_dict(torch.load(modelfile))
    model.eval()

    sidelengths = np.array([get_sidelengths(points) for points in groups_of_points])
    trans_sidelengths_data = transform_sidelengths_data(sidelengths, data_transforms)
    trans_sidelengths_data = torch.from_numpy(trans_sidelengths_data.astype(np.float32))

    with torch.no_grad():
        energies = model(trans_sidelengths_data)
        n_samples = len(energies)
        return energies.numpy().reshape(n_samples, -1)


# TODO: when comparing the ab initio energies, I need to switch back to getting curves for the ab initio potential
def main(i_epoch: int) -> None:
    geometry_tag = "1"
    geometry_func = MAP_GEOMETRY_TAG_TO_FUNCTION[geometry_tag]

    lattice_constants = np.linspace(2.2, 5.0, 2048)
    groups_of_points = [geometry_func(lat_const) for lat_const in lattice_constants]

    training_data_filepath = model_info.get_training_data_filepath()
    data_transforms = model_info.get_data_transforms()
    params = model_info.get_training_parameters(training_data_filepath, data_transforms)

    model = RegressionMultilayerPerceptron(6, 1, params.layers)
    model_filepath = model_info.get_saved_models_dirpath(params) / f"nnpes_{i_epoch:0>5d}.pth"

    analytic_energies = get_analytic_energies(groups_of_points)
    model_energies = get_nn_model_energies(groups_of_points, model, model_filepath, data_transforms)

    fig, ax = plt.subplots()

    ax.set_title(r"Interaction energy for tetrahedron", fontsize=18)
    ax.set_xlabel(r"lattice constant $ / \ \AA $", fontsize=18)
    ax.set_ylabel(r"energy $ / \ \mathrm{cm}^{-1} $", fontsize=18)

    ax.set_xlim(2.0, 6.0)
    ax.set_ylim(-20.0, 220.0)

    ax.set_xticks(np.arange(2.0, 6.001, 0.1), minor=True)
    ax.set_yticks(np.arange(-20.0, 220.01, 40.0), minor=False)
    ax.set_yticks(np.arange(-20.0, 220.01, 8.0), minor=True)

    ax.axhline(y=0, xmin=0.0, xmax=1.0, color="k", lw=1, alpha=0.5)
    ax.plot(lattice_constants, analytic_energies, label="analytic (dummy)")
    ax.plot(lattice_constants, model_energies, label="neural network")

    ax.legend(fontsize=14)

    fig.tight_layout()

    plt.show()
    # plt.savefig("neural_network_energy_example.png", dpi=400)


if __name__ == "__main__":
    main(int(sys.argv[1]))

#    standardized_transformer = StandardizeTransformer((2.2, 10.0), (1.0, 2.0))
#    s = (2.2, 2.5, 2.8, 3.4, 5.5, 10.0)
#    st = standardized_transformer(s)
#
#    print(st)
