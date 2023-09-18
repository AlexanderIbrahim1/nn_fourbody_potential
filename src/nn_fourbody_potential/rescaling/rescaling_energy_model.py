"""
This module contains the RescalingEnergyModel class, which wraps around a trained rescaled model
and a reverse rescaler, and provides a call API that returns the original energy.
"""

import dataclasses

import torch

from nn_fourbody_potential.models import RegressionMultilayerPerceptron

from nn_fourbody_potential.rescaling.reverse_energy_rescaler import ReverseEnergyRescaler


@dataclasses.dataclass
class RescalingEnergyModel:
    rescaled_model: RegressionMultilayerPerceptron
    reverse_rescaler: ReverseEnergyRescaler

    def __call__(self, samples: torch.Tensor, sidelength_groups: torch.Tensor) -> torch.Tensor:
        # NOTE: technically, we can recover the `sidelength_groups` from the `samples` by taking
        # the transformations used to create the samples from the six side lengths, and reversing
        # them; however, in my cases the six side lengths are always available anyways, so I might
        # as well use them
        n_samples = len(samples)

        if n_samples == 0:
            return torch.Tensor([])

        self.rescaled_model.eval()
        with torch.no_grad():
            rescaled_energies: torch.Tensor = self.rescaled_model.forward(samples)

        for i, (eng, sidelengths) in enumerate(zip(rescaled_energies, sidelength_groups)):
            tup_sidelengths = tuple([s.item() for s in sidelengths])
            rescaled_energies[i] = self.reverse_rescaler(eng.item(), tup_sidelengths)

        return rescaled_energies

    def eval(self) -> None:
        pass
