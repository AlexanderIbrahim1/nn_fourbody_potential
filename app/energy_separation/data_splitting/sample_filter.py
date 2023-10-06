"""
This module contains the SampleFilter class, which helps filter the raw sampled ab-initio energies
based on their energies and mean side lengths.
"""

from abc import ABC
from abc import abstractmethod
import dataclasses


import torch


class SampleFilter(ABC):
    @abstractmethod
    def __call__(self, sample_: torch.Tensor) -> bool:
        pass


@dataclasses.dataclass
class FullSampleFilter(SampleFilter):
    min_abs_energy: float
    max_mean_side_length: float

    def __post_init__(self) -> None:
        self.max_side_length_sample_filter = MaxSideLengthSampleFilter(self.max_mean_side_length)
        self.min_energy_sample_filter = MinEnergySampleFilter(self.min_abs_energy)

    def __call__(self, sample_: torch.Tensor) -> bool:
        return self.max_side_length_sample_filter(sample_) and self.min_energy_sample_filter(sample_)


@dataclasses.dataclass
class MaxSideLengthSampleFilter(SampleFilter):
    max_mean_side_length: float

    def __call__(self, sample_: torch.Tensor) -> bool:
        if sample_.shape not in [torch.Size([7]), torch.Size([1, 7])]:
            raise ValueError("Invalid sample passed into the SampleFilter.")

        sample = sample_.view(7)
        side_length_groups = sample[:6]

        mean_side_length = side_length_groups.mean().item()

        return mean_side_length <= self.max_mean_side_length


@dataclasses.dataclass
class MinEnergySampleFilter(SampleFilter):
    min_abs_energy: float

    def __call__(self, sample_: torch.Tensor) -> bool:
        if sample_.shape not in [torch.Size([7]), torch.Size([1, 7])]:
            raise ValueError("Invalid sample passed into the SampleFilter.")

        sample = sample_.view(7)
        energy = sample[6]
        abs_energy = energy.abs().item()

        return abs_energy >= self.min_abs_energy


@dataclasses.dataclass
class MaxEnergySampleFilter(SampleFilter):
    max_abs_energy: float

    def __call__(self, sample_: torch.Tensor) -> bool:
        if sample_.shape not in [torch.Size([7]), torch.Size([1, 7])]:
            raise ValueError("Invalid sample passed into the SampleFilter.")

        sample = sample_.view(7)
        energy = sample[6]
        abs_energy = energy.abs().item()

        return abs_energy < self.max_abs_energy


def apply_filter(samples: torch.Tensor, sample_filter: SampleFilter) -> torch.Tensor:
    if len(samples.shape) != 2 and samples.shape[-1] != 7:
        raise ValueError("Samples of invalid size passed to filter.")

    mask = torch.tensor([sample_filter(s) for s in samples])
    return samples[mask]
