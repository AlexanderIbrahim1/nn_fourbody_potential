from typing import Callable

from nn_fourbody_potential.cartesian import Cartesian3D

PseudoRNG = Callable[[], float]
SixSideLengths = tuple[float, float, float, float, float, float]
SideLengthGenerator = Callable[[], SixSideLengths]
FourCartesianPoints = tuple[Cartesian3D, Cartesian3D, Cartesian3D, Cartesian3D]
