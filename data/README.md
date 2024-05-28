## nn_fourbody_potential_data

This directory contains the data used with the `nn_fourbody_potential` package.
The data and the code are intentionally kept separate.

This repository only contains the isotropic AVDZ *ab initio* four-body interaction energies. These are the energies recovered after performing the Counterpoise Correction on the 15 separate energies recovered from the electronic structure calculations, and then taking 81 of these Counterpoise Corrrected energies and performing a Lebedev quadrature with them.

The raw input and output files involved in the electronic structure calculations are available on [Zenodo](doi:10.5281/zenodo.11272857).
