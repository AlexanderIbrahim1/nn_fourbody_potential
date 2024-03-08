from typing import Annotated
from typing import Sequence
from typing import Tuple

from cartesian import Cartesian3D

FourCartesianPoints = Annotated[Sequence[Cartesian3D], 4]
SixSideLengths = Tuple[float, float, float, float, float, float]
