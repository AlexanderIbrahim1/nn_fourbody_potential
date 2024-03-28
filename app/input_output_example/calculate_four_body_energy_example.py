"""
This script takes an input file 'input.txt', reads the 6 side lengths from within,
and calculates the four-body interaction correction energy between four parahydrogen
molecules with the geometry created from those six side lengths. This energy is stored
in the file 'output.txt'.
"""

from pathlib import Path

import torch
from nn_fourbody_potential import load_potential


def read_six_side_lengths(filepath: Path) -> torch.Tensor:
    with open(filepath, "r") as fin:
        for line in fin:
            if line.startswith("#"):
                continue

            # the only line that doesn't have a comment is the one with the six side lengths
            tokens = line.split()
            return torch.tensor([float(tk) for tk in tokens], dtype=torch.float32)

    assert False, "should never reach this part if you didn't modify the input.txt file"


def main() -> None:
    input_filepath = Path("input.txt")
    side_lengths = read_six_side_lengths(input_filepath).reshape(1, 6)

    size_label = "size64"
    model_filepath = Path("..", "..", "models", "fourbodypara_64_128_128_64.pth")
    device = "cpu"
    potential = load_potential(size_label, model_filepath, device=device)

    energies = potential(side_lengths)

    output_filepath = Path("output.txt")
    with open(output_filepath, "w") as fout:
        fout.write(f"{energies.item(): 12.8f}")


if __name__ == "__main__":
    main()
