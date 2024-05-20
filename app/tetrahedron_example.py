"""
This example shows the use of the fully extrapolated PES for calculating the
four-body interaction potential energy for the tetrahedron geometry, as a function
of the tetrahedron geometry's side length.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from nn_fourbody_potential import load_potential


def plot_tetrahedron_energies(side_lengths: torch.Tensor, energies: torch.Tensor, scale: str) -> None:
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot()

    ax.axhline(y=0, xmin=0, xmax=1, color="k", alpha=0.5, lw=1)
    ax.plot(side_lengths.tolist(), energies.tolist())

    if scale == "full":
        ax.set_xlim(2.0, 5.0)
    elif scale == "zoom":
        ax.set_xlim(3.5, 5.0)
        ax.set_ylim(-0.01, 0.03)
    else:
        assert False, "invalid 'scale' entered: `full` or `zoom`"

    ax.set_title(r"Four-body Interaction Energy of Tetrahedron $ vs $ Sidelength", fontsize=14)
    ax.set_xlabel(r"$ r \ / \ \mathrm{\AA} $", fontsize=14)
    ax.set_ylabel(r"$ V_4(r) \ / \ \mathrm{cm}^{-1} $", fontsize=14)

    fig.tight_layout()
    plt.show()


def main(device: str, scale: str) -> None:
    assert scale in ["full", "zoom"]

    # select the model size ("size8", "size16", "size32", "size64"), and the path to that model
    size_label = "size64"

    # select the activation label ("relu", "shiftedsoftplus")
    # - note that "shiftedsoftplus" only works if `size_label == "size64"`
    activation_label = "shiftedsoftplus"

    # create the path to the model to load
    model_filepath = Path("..", "models", "fourbodypara_ssp_64_128_128_64.pth")

    # create the fully-extrapolated PES
    potential = load_potential(size_label, activation_label, model_filepath, device=device)

    # create perfect tetrahedron geometries with side lengths between 2.0 and 5.0 Angstroms
    side_lengths = torch.linspace(2.0, 5.0, 1024)

    # the model takes its inputs as 6-tuples of the geometry side lengths
    input_sidelengths = torch.tensor([[r for _ in range(6)] for r in side_lengths])
    input_sidelengths = input_sidelengths.reshape(1024, 6)

    # move the input onto the device
    input_sidelengths = input_sidelengths.to(device)

    # get the energies as a 1D torch Tensor
    energies: torch.Tensor = potential(input_sidelengths)

    # plot the energies against the side lengths
    plot_tetrahedron_energies(side_lengths, energies, scale)


if __name__ == "__main__":
    # set `device` to either "cpu" or "cuda", depending on where the neural network
    # calculations will take place
    device: str = "cpu"

    # set `scale` to "full" or "zoom" to view a plot of the four-body interaction energies
    # as a function of side length;
    # - "full" means the energies between 2.0 and 5.0 angstroms will be shown
    # - "zoom" means the plot will zoom into the weak energy dip around 3.7 angstroms, as
    #   shown in the publication that accompanies this repository
    scale: str = "full"
    main(device, scale)
