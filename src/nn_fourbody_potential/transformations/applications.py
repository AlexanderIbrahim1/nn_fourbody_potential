"""
This module contains functions for applying a series of transformations to a collection
of six sidelengths.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Sequence

import numpy as np

from nn_fourbody_potential.transformations.transformers import SixSideLengthsTransformer
from nn_fourbody_potential.transformations.transformers import SixSideLengths
from nn_fourbody_potential.transformations.transformers import TransformedSideLengths


def apply_transformations(
    sidelengths: SixSideLengths,
    data_transforms: Sequence[SixSideLengthsTransformer],
) -> Sequence[TransformedSideLengths]:
    """Apply all the data transformations, in order, to the six side lengths."""
    trans_sidelengths = deepcopy(sidelengths)
    for transform in data_transforms:
        trans_sidelengths = transform(trans_sidelengths)

    return trans_sidelengths


def apply_transformations_to_sidelengths_data(
    sidelengths: np.ndarray[float, float],
    data_transforms: Sequence[SixSideLengthsTransformer],
) -> np.ndarray[float, float]:
    """
    Apply the 'apply_transformations()' method to a sequence of SixSideLengths instances.

    In the cases of interest to this project, the 'sequence of SixSideLengths instances' is
    a 2D numpy array of shape (n_samples, 6).
    """
    trans_sidelengths = np.empty(sidelengths.shape, dtype=sidelengths.dtype)

    for (n_sample, sidelens) in enumerate(sidelengths):
        trans_sidelengths[n_sample] = apply_transformations(sidelens, data_transforms)

    return trans_sidelengths
