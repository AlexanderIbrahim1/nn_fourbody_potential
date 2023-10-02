"""
This module contains files that make the splitting of the data into training, testing,
and validation sets more straightforward and reproducible.
"""

import dataclasses
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import random_split

from sample_filter import SampleFilter
from sample_filter import apply_filter
from sample_splitter import DataSplitFractions
from sample_splitter import DataSplitSizes
from sample_splitter import fractions_to_sizes

# PLAN
# [DONE] - set the seeds up front, in both torch and numpy
# [DONE] - set the splitting percentages up front
# [DONE] - create the filter up front
#
# [DONE] - load the unfiltered hcp and sampled data
# [DONE] - create filtered hcp and sampled data
# [DONE] - split the sampled data into training wo/hcp, testing, and validation
# [DONE] - create a new training set that combines the sampled training data with the hcp data
# [DONE] - save all the data in a new directory


@dataclasses.dataclass
class FilePaths:
    hcp_filepath: Path
    sampled_filepath: Path
    train_save_filepath: Path
    train_nohcp_save_filepath: Path
    test_save_filepath: Path
    valid_save_filepath: Path


def main(
    seed: int, fractions: DataSplitFractions, data_filter: Callable[[torch.Tensor], bool], filepaths: FilePaths
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    hcp_unfiltered = torch.from_numpy(np.loadtxt(filepaths.hcp_filepath)).view(-1, 7)
    sampled_unfiltered = torch.from_numpy(np.loadtxt(filepaths.sampled_filepath)).view(-1, 7)

    hcp_data = apply_filter(hcp_unfiltered, data_filter)
    sampled_data = apply_filter(sampled_unfiltered, data_filter)

    sizes: DataSplitSizes = fractions_to_sizes(sampled_data.shape[0], fractions)
    splits = random_split(sampled_data, [sizes.train, sizes.test, sizes.valid])

    train_nohcp_data = torch.stack([sampled_data[i] for i in splits[0].indices]).numpy()
    test_data = torch.stack([sampled_data[i] for i in splits[1].indices]).numpy()
    valid_data = torch.stack([sampled_data[i] for i in splits[2].indices]).numpy()
    train_data = np.concatenate((hcp_data, train_nohcp_data))

    np.savetxt(filepaths.train_nohcp_save_filepath, train_nohcp_data)
    np.savetxt(filepaths.train_save_filepath, train_data)
    np.savetxt(filepaths.test_save_filepath, test_data)
    np.savetxt(filepaths.valid_save_filepath, valid_data)


if __name__ == "__main__":
    seed = 42
    fractions = DataSplitFractions(0.7, 0.15, 0.15)
    data_filter = SampleFilter(1.0e-3, 4.5)
    filepaths = FilePaths(
        hcp_filepath=Path("..", "data", "abinitio_hcp_data_3901.dat"),
        sampled_filepath=Path("..", "data", "abinitio_sampled_data_16000.dat"),
        train_save_filepath=Path(".", "split_data", "train2.dat"),
        train_nohcp_save_filepath=Path(".", "split_data", "train_nohcp2.dat"),
        test_save_filepath=Path(".", "split_data", "test2.dat"),
        valid_save_filepath=Path(".", "split_data", "valid2.dat"),
    )

    main(seed, fractions, data_filter, filepaths)
