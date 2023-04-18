from typing import Tuple

from torch.utils.data import Dataset

from torchtyping import TensorType
from torchtyping import patch_typeguard
from typeguard import typechecked


@typechecked
class PotentialDataset(Dataset):
    """
    Prepare the dataset of the mapping of a multidimensional feature space to a
    single output dimension.
    """

    def __init__(
        self,
        x: TensorType["samples", "features"],
        y: TensorType["samples", "outputs"],
    ) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index) -> Tuple[TensorType["features"], TensorType["outputs"]]:
        return (self.x[index], self.y[index])
