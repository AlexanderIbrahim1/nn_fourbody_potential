"""
This module contains an example of using the neural network potential energy surface. This
only exists to show the "guts" of what the inputs and outputs of the neural network go
through. For the full working potential, see `example.py` in this directory.

NOTE: this example only gives reasonable outputs for geometries that are in the same subset
of coordinate space as the training data. It is the job of the ExtrapolatedPotential class
to handle geometries that have very large or very small side lengths.

---

It begins with the four centres of mass (points in 3D Cartesian space), one for each
parahydrogen molecule, and calculates the six relative pair distances from them. These
six relative pair distances are fed into the feature transformers, then into the neural
network.

The output of the neural network is a rescaled version of the four-body interaction energy.
To get the true energy, the reverse rescaling function needs to be created, and called with
the neural network's output.

The six relative pair distances can come from any source; there is no need to start with the
four points in 3D Cartesian space.

The input side lengths (before the transformations are applied) are in units of Angstroms.
The output energies are in units of wavenumbers.
"""

import numpy as np
import torch

from pathlib import Path

from nn_fourbody_potential.cartesian import Cartesian3D
from nn_fourbody_potential.cartesian import relative_pair_distances
from nn_fourbody_potential.constants import ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_COEFF
from nn_fourbody_potential.constants import ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_EXPON
from nn_fourbody_potential.constants import N_FEATURES
from nn_fourbody_potential.constants import N_OUTPUTS
from nn_fourbody_potential.dispersion4b import b12_parahydrogen_midzuno_kihara
from nn_fourbody_potential.models import RegressionMultilayerPerceptron
from nn_fourbody_potential.transformations import SixSideLengthsTransformer
from nn_fourbody_potential.transformations import ReciprocalTransformer
from nn_fourbody_potential.transformations import MinimumPermutationTransformer
from nn_fourbody_potential.transformations import StandardizeTransformer
from nn_fourbody_potential.transformations import transform_sidelengths_data
from nn_fourbody_potential import rescaling


def feature_transformers() -> list[SixSideLengthsTransformer]:
    min_sidelen = 2.2

    return [
        ReciprocalTransformer(),
        StandardizeTransformer((0.0, 1.0 / min_sidelen), (0.0, 1.0)),
        MinimumPermutationTransformer(),
    ]


def output_rescaling_function() -> rescaling.RescalingFunction:
    # constants chosen by hand, to drastically decrease the dynamic range of the output data;
    # after a certain amount of fiddling, it seems several choices of parameters would be good enough
    coeff = ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_COEFF / 12.0
    expon = ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_EXPON * 5.02
    disp_coeff = 0.125 * b12_parahydrogen_midzuno_kihara()

    return rescaling.RescalingFunction(coeff, expon, disp_coeff)


def output_to_energy_rescaler() -> rescaling.ReverseEnergyRescaler:
    rescaling_function = rescaling_function()

    # the rescaling values determined from the training data
    rescaling_limits_to_energies = rescaling.RescalingLimits(
        from_left=-1.0, from_right=1.0, to_left=-3.2619903087615967, to_right=8.64592170715332
    )
    rescaler_output_to_energies = rescaling.ReverseEnergyRescaler(rescaling_function, rescaling_limits_to_energies)

    return rescaler_output_to_energies


def main(*, device: str) -> None:
    # begin with the four points in 3D Cartesian space
    # the values chosen correspond to (almost) a tetrahedron with a side length of 2.2 Angstroms
    points = [
        Cartesian3D(0.000, 0.000, 0.000),
        Cartesian3D(2.200, 0.000, 0.000),
        Cartesian3D(1.100, 1.905, 0.000),
        Cartesian3D(1.100, 0.635, 1.796),
    ]

    # get the six relative pair distances (r01, r02, r03, r12, r13, r23)
    pair_distances = relative_pair_distances(points)

    # convert to a 2D numpy array
    # the first dimension is the number of samples in the batch (1 in this case), to comply with pytorch
    # the second dimension corresponds to the number of side lengths
    n_samples = 1
    n_sidelengths = 6
    pair_distances = np.array(pair_distances).reshape(n_samples, n_sidelengths)

    # get the functions that transform the six pair distances
    transformers = feature_transformers()

    # apply the transformations to the six pair distances
    # then convert the result to a torch Tensor
    input_data = transform_sidelengths_data(pair_distances, transformers)
    input_data = torch.from_numpy(input_data.astype(np.float32)).to(device)

    # create the PyTorch model; the following parameters (input features, outputs, and the layer sizes)
    # are specific to the model that was trained;
    # so far, the weights have not been initialized
    layer_sizes = [64, 128, 128, 64]
    model = RegressionMultilayerPerceptron(N_FEATURES, N_OUTPUTS, layer_sizes)

    # the path to the specific .pth file
    modelfile = Path("..", "models", "fourbodypara_64_128_128_64.pth")

    # fill the weights of the model
    model_state_dict = torch.load(modelfile, map_location=torch.device(device))
    model.load_state_dict(model_state_dict)
    model = model.to(device)

    # the model should be put in evaluation mode, and the gradients should be turned off
    with torch.no_grad():
        model.eval()
        output_data: torch.Tensor = model(input_data)

    rescaling_function = output_rescaling_function()

    # linear rescaling limits (determined from training data for these models)
    rescaling_limits_to_energies = rescaling.RescalingLimits(
        from_left=-1.0, from_right=1.0, to_left=-3.2619903087615967, to_right=8.64592170715332
    )

    # the functor that takes the neural network output and converts it back to the energy
    rescaler_output_to_energies = rescaling.ReverseEnergyRescaler(rescaling_function, rescaling_limits_to_energies)

    # the rescaling requires both the neural network output and the original six side lengths
    interaction_energy = rescaler_output_to_energies(output_data.item(), pair_distances.flatten().tolist())

    print(f"The interaction energy is {interaction_energy} cm^{{-1}}.")


if __name__ == "__main__":
    device = "cpu"
    main(device=device)
