"""
This module contains components for generating the training data for the four-body
potential energy surface.

The training data is made up of the 6 relative side lengths of a four-body geometry,
and the corresponding energy value.
"""

from __future__ import annotations

from pathlib import Path

from nn_fourbody_potential.fourbody_potential import create_fourbody_analytic_potential
from nn_fourbody_potential.sidelength_distributions import get_abinit_tetrahedron_distribution
from nn_fourbody_potential.sidelength_distributions import generate_training_data

from nn_fourbody_potential.dataio import save_fourbody_training_data


def main() -> None:
    n_samples = 5000
    distrib = get_abinit_tetrahedron_distribution()
    potential = create_fourbody_analytic_potential()
    filename = Path("data", f"training_data_{n_samples}.dat")

    sidelengths, energies = generate_training_data(n_samples, distrib, potential)
    save_fourbody_training_data(filename, sidelengths, energies)


if __name__ == "__main__":
    main()
