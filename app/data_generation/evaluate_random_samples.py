"""
In this script, we create the fully extrapolated PES, and evaluate randomly generated
samples.
"""

from pathlib import Path

import torch

from common_types import SideLengthGenerator
from common_types import SixSideLengths
from random_generation import make_prng
from random_generation import MAP_CATEGORY_TO_SIDE_LENGTH_GENERATOR
from side_lengths_to_cartesian import maybe_six_side_lengths_to_cartesian

from nn_fourbody_potential import load_potential


def generate_n_valid_samples(
    generator: SideLengthGenerator, n_samples: int, n_max_attempts_per_sample: int = 1000
) -> list[SixSideLengths]:
    output: list[SixSideLengths] = []

    for _ in range(n_samples):
        for i_attempt in range(n_max_attempts_per_sample):
            sample = generator()
            side_lengths = maybe_six_side_lengths_to_cartesian(*sample)
            if side_lengths is not None:
                output.append(sample)
                break

            if i_attempt == n_max_attempts_per_sample - 1:
                raise RuntimeError("Failed too many attempts to generate a valid sample.")

    return output


def main(device: str, generator: SideLengthGenerator, n_samples: int) -> None:
    size_label = "size64"
    model_filepath = Path("..", "..", "models", "fourbodypara_64_128_128_64.pth")
    potential = load_potential(size_label, model_filepath, device=device)

    input_side_lengths = generate_n_valid_samples(generator, n_samples)
    input_side_lengths = torch.Tensor(input_side_lengths)
    input_side_lengths = input_side_lengths.reshape(n_samples, 6)
    input_side_lengths = input_side_lengths.to(device)

    # get the energies as a 1D torch Tensor
    energies: torch.Tensor = potential(input_side_lengths)

    print(energies)


if __name__ == "__main__":
    device: str = "cpu"
    n_samples: int = 1000

    short = make_prng(1.5, 2.0)
    long = make_prng(3.0, 5.0)

    generator_maker = MAP_CATEGORY_TO_SIDE_LENGTH_GENERATOR[5]
    generator = generator_maker(short, long)

    main(device, generator, n_samples)
