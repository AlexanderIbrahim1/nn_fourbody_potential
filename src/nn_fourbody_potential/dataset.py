from typing import Tuple

from torch.utils.data import Dataset


class PotentialDataset(Dataset):
    """
    Prepare the dataset of the mapping of a multidimensional feature space to a
    single output dimension.
    """

    def __init__(
        self,
        x,
        y,
    ) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index):
        return (self.x[index], self.y[index])
