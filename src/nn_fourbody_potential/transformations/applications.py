"""
This module contains functions for applying a series of transformations to a collection
of six sidelengths.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from nn_fourbody_potential.transformations.transformers import SixSideLengthsTransformer
from nn_fourbody_potential.common_types import SixSideLengths
from nn_fourbody_potential.common_types import TransformedSideLengths


def transform_sidelengths_data(
    sidelengths: NDArray, transforms: Sequence[SixSideLengthsTransformer]
) -> np.ndarray[float, float]:
    """Apply the 'apply_transformations()' method to a sequence of SixSideLengths instances."""
    return np.array([_apply_transformations(sidelens, transforms) for sidelens in sidelengths])


def _apply_transformations(
    sidelengths: SixSideLengths,
    data_transforms: Sequence[SixSideLengthsTransformer],
) -> Sequence[TransformedSideLengths]:
    """Apply all the data transformations, in order, to the six side lengths."""
    trans_sidelengths = deepcopy(sidelengths)
    for transform in data_transforms:
        trans_sidelengths = transform(trans_sidelengths)

    return trans_sidelengths
