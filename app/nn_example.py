"""
This module contains an example of using the neural network potential energy surface.

It begins with the four centres of mass (points in 3D Cartesian space), one for each
parahydrogen molecule, and calculates the six relative pair distances from them. These
six relative pair distances are fed into the feature transformers, then into the neural
network.

The six relative pair distances can come from any source; there is no need to start with the
four points in 3d Cartesian space.

The input side lengths (before the transformations are applied) are in units of Angstroms.
The output energies are in units of wavenumbers.
"""

import numpy as np
import torch

from pathlib import Path

from nn_fourbody_potential.cartesian import Cartesian3D
from nn_fourbody_potential.cartesian import relative_pair_distances
from nn_fourbody_potential.models import RegressionMultilayerPerceptron
from nn_fourbody_potential.transformations import SixSideLengthsTransformer
from nn_fourbody_potential.transformations import ReciprocalTransformer
from nn_fourbody_potential.transformations import MinimumPermutationTransformer
from nn_fourbody_potential.transformations import StandardizeTransformer
from nn_fourbody_potential.transformations import transform_sidelengths_data


def feature_transformers() -> list[SixSideLengthsTransformer]:
    min_sidelen = 2.2
    max_sidelen = 4.5

    return [
        ReciprocalTransformer(),
        StandardizeTransformer((1.0 / max_sidelen, 1.0 / min_sidelen), (0.0, 1.0)),
        MinimumPermutationTransformer(),
    ]


def main() -> None:
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
    input_data = torch.from_numpy(input_data.astype(np.float32))

    # create the PyTorch model; the following parameters (input features, outputs, and the layer sizes)
    # are specific to the model that was trained;
    # so far, the weights have not been initialized
    n_features = 6
    n_outputs = 1
    layer_sizes = [64, 128, 128, 64]
    model = RegressionMultilayerPerceptron(n_features, n_outputs, layer_sizes)

    # the path to the specific .pth file
    # 2999 corresponds to the very last batch
    modelfile = Path(
        "models", "nnpes_initial_layers64_128_128_64_lr_0.000200_datasize_8901", "models", "nnpes_02999.pth"
    )

    # fill the weights of the model
    model.load_state_dict(torch.load(modelfile))

    # the model should be put in evaluation mode, and the gradients should be turned off
    with torch.no_grad():
        model.eval()
        output_data = model(input_data)

        # the output of the model is a 1x1 torch Tensor; we must extract the floating-point value from it
        interaction_energy = output_data.item()

        print(f"The interaction energy is {interaction_energy} cm^{{-1}}.")


if __name__ == "__main__":
    main()
