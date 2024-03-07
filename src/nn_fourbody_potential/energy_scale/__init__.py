"""
The EnergyScaleEnsembleModel was an earlier attempt at creating the potential energy surface.

Several different models were trained, each on a different subset of the training data, based
on the energies of the samples. During evaluation, a sample would be passed to a different model,
based on whether that sample was suspected of belonging to that model's energy regime.

For example, if a sample was suspected of being a high-energy sample, the EnergyScaleEnsembleModel
would pass it along to the high-energy neural network responsible for high-energy samples.

This approach turned out to be much less effective than the energy scaling approach, and so we
abandoned it.

This subpackage is no longer in use.
"""
