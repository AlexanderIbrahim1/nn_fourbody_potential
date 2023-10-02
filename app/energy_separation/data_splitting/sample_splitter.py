"""
This module contains components to help split the data into separate training, testing, and
validation groups.
"""

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class DataSplitFractions:
    train: float
    test: float
    valid: float


@dataclasses.dataclass(frozen=True)
class DataSplitSizes:
    train: int
    test: int
    valid: int


def fractions_to_sizes(total_size: int, fractions: DataSplitFractions) -> DataSplitSizes:
    train_size = np.ceil(total_size * fractions.train).astype(int)
    test_size = np.ceil(total_size * fractions.test).astype(int)
    valid_size = total_size - train_size - test_size

    return DataSplitSizes(train_size, test_size, valid_size)
