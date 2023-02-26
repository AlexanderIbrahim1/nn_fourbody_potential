"""
This module contains code to plot the potential energy surfaces (PES) along paths in
coordinate space. The PESs can be a NN PES, or the analytic PES.
"""

from __future__ import annotations

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
from nn_fourbody_potential.transformations import MinimumPermutationTransformer
from nn_fourbody_potential.transformations import ReciprocalTransformer
from nn_fourbody_potential.transformations import StandardizeTransformer
from nn_fourbody_potential.transformations import apply_transformations


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

    n_sidelengths = 6
    n_samples = len(groups_of_points)

    trans_sidelengths_data = np.empty((n_samples, n_sidelengths), dtype=float)
    for (i_sample, points) in enumerate(groups_of_points):
        sidelengths = get_sidelengths(points)
        trans_sidelengths_data[i_sample] = apply_transformations(
            sidelengths, data_transforms
        )
    trans_sidelengths_data = torch.from_numpy(trans_sidelengths_data.astype(np.float32))

    with torch.no_grad():
        energies = model(trans_sidelengths_data)

        return energies.numpy().reshape(n_samples, -1)


def main() -> None:
    geometry_tag = "1"
    geometry_func = MAP_GEOMETRY_TAG_TO_FUNCTION[geometry_tag]

    lattice_constants = np.linspace(2.2, 6.0, 256)
    groups_of_points = [
        geometry_func(lat_const) for lat_const in lattice_constants
    ]

    model = RegressionMultilayerPerceptron(6, 1, [64, 128, 128, 64])
    modelfile = Path(
        ".",
        "models",
        "nn_pes_model_64_128_128_64_minpermute_reciprocal_lr0005_epoch500_batch1000_wdecay_00001_data5000_2.2to5.0.pth",
    )
    data_transforms = [MinimumPermutationTransformer(), ReciprocalTransformer()]

    analytic_energies = get_analytic_energies(groups_of_points)
    model_energies = get_nn_model_energies(
        groups_of_points, model, modelfile, data_transforms
    )

    fig, ax = plt.subplots()
    ax.plot(lattice_constants, analytic_energies, label='analytic')
    ax.plot(lattice_constants, model_energies, label='model')
    plt.show()

if __name__ == "__main__":
    
    standardized_transformer = StandardizeTransformer((2.2, 10.0), (1.0, 2.0))
    s = (2.2, 2.5, 2.8, 3.4, 5.5, 10.0)
    st = standardized_transformer(s)
    
    print(st)