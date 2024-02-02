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

from sample_filter import MaxSideLengthSampleFilter
from sample_filter import apply_filter
from sample_splitter import DataSplitFractions
from sample_splitter import DataSplitSizes
from sample_splitter import fractions_to_sizes

import script_utils


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
    splits = random_split(sampled_data, [sizes.train, sizes.test, sizes.valid])  # type: ignore

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
    fractions = DataSplitFractions(0.75, 0.125, 0.125)
    data_filter = MaxSideLengthSampleFilter(4.5)
    filepaths = FilePaths(
        hcp_filepath=script_utils.RAW_ABINITIO_HCP_DATA_FILEPATH,
        sampled_filepath=script_utils.RAW_ABINITIO_ALL_SAMPLED_DATA_FILEPATH,
        train_save_filepath=script_utils.FILTERED_SPLIT_ABINITIO_TRAIN_DATA_DIRPATH,
        train_nohcp_save_filepath=script_utils.FILTERED_SPLIT_ABINITIO_TRAIN_NOHCP_DATA_DIRPATH,
        test_save_filepath=script_utils.FILTERED_SPLIT_ABINITIO_TEST_DATA_DIRPATH,
        valid_save_filepath=script_utils.FILTERED_SPLIT_ABINITIO_VALID_DATA_DIRPATH,
    )

    main(seed, fractions, data_filter, filepaths)
