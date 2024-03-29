# nn_fourbody_potential

This is the repository for the potential energy surface (PES) for the four-body correction interaction energy between four molecules of parahydrogen. Only the isotopic term in the potential expansion is used.

We recommend using the potential through Python (described in the `Recommended Usage` section below).

For compliance with the requirements for publishing PESs in the Journal of Chemical Physics, we also provide an example where a Python script reads an input file and produces an output file (described in the `Input and Output File Example` section below).


## Requirements
```bash
Python 3.9 <= 3.11
torch >= 2.0.0
matplotlib >= 3.8.3   # for seeing the plot in `example.py`
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

size_label: str = "size64"
model_filepath: Path = Path('path', 'to', 'pytorch', 'model', 'model.pth')
device: str = "cpu"  # "cuda"

potential = load_potential(size_label, model_filepath, device=device)

# six side lengths go here
side_lengths = torch.tensor([2.2, 2.3, 2.4, 2.5, 2.45, 2.35]).reshape(1, 6).to(device)
energy = potential(side_lengths)
```

The files `app/tetrahedron_example.py` and `app/input_output_example/calculate_four_body_energy_example.py` provide examples of using the potential this way.


## Input and Output File Example
In the directory `app/input_output_example`, we provide a Python script `calculate_four_body_energy_example.py` and a file `input.txt`.

The `input.txt` file contains an example of six side lengths of a four-body geometry.
The Python script will read in those six side lengths, calculate the four-body correction interaction energy for four parahydrogen molecules, and write them to `output.txt`.


## Data Cleaning and Training
The code for splitting the data into training, testing, and validation sets is located in `app/data_cleaning`. The main script of interest is `app/data_cleaning/split_energies.py`.

The code for training the neural network is located in `app/training`.

The training is done through the `train_fourbody_model()` function in `app/training/training.py`. The other modules in the `app/training` contain functions and types that assist with the training process. The code is currently set for training the `64-128-128-64` model.
