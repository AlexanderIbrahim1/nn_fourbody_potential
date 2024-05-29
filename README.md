# nn_fourbody_potential

**NOTE: users who have accessed this repository through the `PES.tar` file provided as supplementary material by the Journal of Chemical Physics, are strongly encouraged to access the github repository [github.com/AlexanderIbrahim1/nn_fourbody_potential](https://github.com/AlexanderIbrahim1/nn_fourbody_potential) for any possible updates to the potential.**

This is the repository for the reference implementation of the potential energy surface (PES) for the four-body non-additive interaction energy between four molecules of parahydrogen. Only the isotopic term in the potential expansion is used.

We recommend using the potential through Python (described in the `Recommended Usage` section below).

For compliance with the requirements for publishing PESs in the Journal of Chemical Physics, we also provide an example where a Python script reads an input file and produces an output file (described in the `Input and Output File Example` section below).


## Requirements
```bash
Python 3.9 <= 3.11
torch >= 2.0.0
matplotlib >= 3.8.3   # for seeing the plot in `tetrahedron_example.py`
```


## Instructions
1a. If accessing from github, clone the repo and enter
```bash
git clone git@github.com:AlexanderIbrahim1/nn_fourbody_potential.git
cd nn_fourbody_potential
```
1b. If accessing from the `PES.tar` file provided by JCP, untar and enter
```bash
tar -xvf PES.tar
cd nn_fourbody_potential
```

2. Create the virtual environment (unless you already have one set up)
```bash
virtualenv venv -p python  # version 3.9, 3.10, or 3.11 as of time of release
```

3. Install the package
```bash
pip install -r requirements.txt
pip install .
cd data  # training, testing, validation data stored separately from the source code
pip install .
cd ..
```

4. Run the examples
```bash
cd app
python tetrahedron_example.py  # get four-body interaction energy for the tetrahedron geometry
python nn_example.py  # the neural network without the ExtrapolatedPotential wrapper
```


## Recommended Usage
We recommend creating and using the potential in a way similar to the following
```py
from pathlib import Path
from nn_fourbody_potential import load_potential
import torch

# an `activation_label` of "relu" works with `size_label` values of "size8", "size16", "size32", or "size64"
# an `activation_label` of "shiftedsoftplus" only works with a `size_label` value of "size64"
size_label: str = "size64"
activation_label: str = "shiftedsoftplus"
model_filepath: Path = Path("path", "to", "pytorch", "model", "model.pth")
device: str = "cpu"  # "cuda"
potential = load_potential(size_label, activation_label, model_filepath, device=device)

# six side lengths go here
side_lengths = torch.tensor([2.2, 2.3, 2.4, 2.5, 2.45, 2.35]).reshape(1, 6).to(device)
energy = potential(side_lengths)
```

The files `app/tetrahedron_example.py` and `app/input_output_example/calculate_four_body_energy_example.py` provide examples of using the potential this way.

The `load_potential()` function also accepts the optional keyword flag `use_lookupshortest_permutation: bool`. When set to `True`, it uses the version of the minimum permutation transformer that assumes that the two shortest side lengths are unique, as described in the paper.


## Input and Output File Example
In the directory `app/input_output_example`, we provide a Python script `calculate_four_body_energy_example.py` and a file `input.txt`.

The `input.txt` file contains an example of six side lengths of a four-body geometry.
The Python script will read in those six side lengths, calculate the isotropic four-body non-additive interaction energy for the four parahydrogen molecules, and write them to `output.txt`.


## Data Cleaning and Training
The code for splitting the data into training, testing, and validation sets is located in `app/data_cleaning`. The main script of interest is `app/data_cleaning/split_energies.py`.

The code for training the neural network is located in `app/training`.

The training is done through the `train_fourbody_model()` function in `app/training/training.py`. The other modules in the `app/training` contain functions and types that assist with the training process. The code is currently set for training the `64-128-128-64-SSP` model.


## Data Availability
The data (side lengths and corresponding isotropic four-body non-additive interaction energies) are located in the `data` directory.

This repository only contains the *isotropic* four-body interaction energies, for which the Counterpoise Correction and the Lebedev quadrature have already been applied. The raw input and output files involved in the electronic structure calculations are available on the scientific data respository [Zenodo](https://zenodo.org/doi/10.5281/zenodo.11272855).


## Two-Body and Three-Body Potentials
The journal publication that accompanies this repository also discusses using this repository alongside a two-body and three-body PESs for parahydrogen.

### Two-Body Potential
The two-body potential is described in the paper **M. Schmidt *et al.* (2015). Raman vibrational shifts of small clusters of hydrogen isotopologues. *J. Phys. Chem. A* vol. 199, p. 12551-12561.**

The function used to construct it is given by equations (15) through (21), using the fit parameters in table (6) that correspond to parahydrogen.

### Three-Body Potential
The three-body potential is described in the paper **A. Ibrahim and P.-N. Roy (2022). Three-body potential energy surface for para-hydrogen. *J. Chem. Phys.* vol. 156, p. 044301.**

The repository used to construct the three-body PES is found at [github.com/roygroup/threebodyparah2](https://github.com/roygroup/threebodyparah2). It contains detailed instructions on how to compile the PES and test it against certain cases. 
