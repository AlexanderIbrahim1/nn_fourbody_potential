"""
This example shows the use of the fully extrapolated PES.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from nn_fourbody_potential import load_potential


def main() -> None:
    size_label = "size64"
    model_filepath = Path("..", "models", "fourbodypara_64_128_128_64.pth")
    potential = load_potential(size_label, model_filepath)

    side_lengths = torch.linspace(2.0, 5.0, 1024)

    input_sidelengths = torch.tensor([[r for _ in range(6)] for r in side_lengths]).reshape(1024, 6).to("cuda")
    energies = potential(input_sidelengths)

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.plot(side_lengths.tolist(), energies.tolist())
    plt.show()


if __name__ == "__main__":
    main()
