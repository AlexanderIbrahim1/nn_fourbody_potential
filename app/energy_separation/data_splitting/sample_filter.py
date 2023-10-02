"""
This module contains the SampleFilter class, which helps filter the raw sampled ab-initio energies
based on their energies and mean side lengths.
"""

import dataclasses

import torch


@dataclasses.dataclass(frozen=True)
class SampleFilter:
    min_abs_energy: float
    max_mean_side_length: float

    def __call__(self, sample_: torch.Tensor) -> bool:
        if sample_.shape not in [torch.Size([7]), torch.Size([1, 7])]:
            raise ValueError("Invalid sample passed into the SampleFilter.")

        sample = sample_.view(7)
        side_length_groups = sample[:6]
        energy = sample[6]

        mean_side_length = side_length_groups.mean().item()
        abs_energy = energy.abs().item()

        return abs_energy >= self.min_abs_energy and mean_side_length <= self.max_mean_side_length


def apply_filter(samples: torch.Tensor, sample_filter: SampleFilter) -> torch.Tensor:
    if len(samples.shape) != 2 and samples.shape[-1] != 7:
        raise ValueError("Samples of invalid size passed to filter.")

    mask = torch.tensor([sample_filter(s) for s in samples])
    return samples[mask]
