from typing import Annotated
from typing import Callable
from typing import Sequence
from typing import Tuple

from nn_fourbody_potential.cartesian import Cartesian3D

FourCartesianPoints = Annotated[Sequence[Cartesian3D], 4]
Permutation = Tuple[int, int, int, int, int, int]
SixSideLengths = Tuple[float, float, float, float, float, float]
SixSideLengthsComparator = Callable[[SixSideLengths, SixSideLengths], bool]
TransformedSideLengths = Tuple[float, float, float, float, float, float]
