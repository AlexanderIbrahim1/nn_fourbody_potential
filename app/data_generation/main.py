"""
This script contains functions for generating the side lengths of samples used in the
model training process.
"""

from common_types import SideLengthGenerator
from common_types import SixSideLengths

from random_generation import make_prng
from random_generation import MAP_CATEGORY_TO_SIDE_LENGTH_GENERATOR

from side_lengths_to_cartesian import maybe_six_side_lengths_to_cartesian

# PLAN
# - create functor that generates a random number
#   - default to uniform distribution
#   - pick two different ranges (short and long)
# - create 7 different functions that return samples with side lengths
#   generated using these side lengths
#   - one function for each category I want to add extra samples in


def generate_n_valid_samples(
    generator: SideLengthGenerator, n_samples: int, n_max_attempts_per_sample: int = 10
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


if __name__ == "__main__":
    short = make_prng(2.2, 2.6)
    long = make_prng(3.3, 5.5)

    generator_maker = MAP_CATEGORY_TO_SIDE_LENGTH_GENERATOR[2]
    generator = generator_maker(short, long)

    side_lengths = generate_n_valid_samples(generator, 1000)
    points = [maybe_six_side_lengths_to_cartesian(*sl) for sl in side_lengths]

    print(points)
