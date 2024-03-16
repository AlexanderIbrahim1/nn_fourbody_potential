# nn_fourbody_potential

A repository for creating the potential energy surface for the interaction energy between four molecules of parahydrogen, using a Multilayer Perceptron, with pytorch.


## Instructions

1. Clone the repo and enter
```bash
git clone git@github.com:AlexanderIbrahim1/nn_fourbody_potential.git
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
python example.py  # get four-body interaction energy for the tetrahedron geometry of several sizes
python nn_example.py  # the neural network without the ExtrapolatedPotential wrapper
```


## Requirements
```bash
Python 3.9 <= 3.11
torch >= 2.0.0
matplotlib >= 3.8.3   # for seeing the plot in `example.py`
```
