from typing import Annotated
from typing import Callable
from typing import Sequence
from typing import Tuple

from nn_fourbody_potential.cartesian import Cartesian3D

FourCartesianPoints = Annotated[Sequence[Cartesian3D], 4]
SixSideLengths = Tuple[float, float, float, float, float, float]
TransformedSideLengths = Tuple[float, float, float, float, float, float]
SixSideLengthsComparator = Callable[[SixSideLengths, SixSideLengths], bool]
