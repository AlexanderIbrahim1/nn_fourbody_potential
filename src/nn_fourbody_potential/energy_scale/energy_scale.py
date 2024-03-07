import enum
from dataclasses import dataclass

from nn_fourbody_potential.full_range.utils import smooth_01_transition


class EnergyScale(enum.Enum):
    LOW = enum.auto()
    MIXED_LOW_MEDIUM = enum.auto()
    MEDIUM = enum.auto()
    MIXED_MEDIUM_HIGH = enum.auto()
    HIGH = enum.auto()


@dataclass(frozen=True)
class EnergyScaleFraction:
    scale: EnergyScale
    lower_fraction: float
    upper_fraction: float


class EnergyScaleAssigner:
    def __init__(
        self,
        *,
        low_medium_centre: float,
        low_medium_width: float,
        medium_high_centre: float,
        medium_high_width: float,
    ) -> None:
        self._check_width_is_nonnegative(low_medium_width)
        self._check_width_is_nonnegative(medium_high_width)

        self._low_medium_lower_end = low_medium_centre - low_medium_width
        self._low_medium_upper_end = low_medium_centre + low_medium_width
        self._medium_high_lower_end = medium_high_centre - medium_high_width
        self._medium_high_upper_end = medium_high_centre + medium_high_width

        self._check_order_of_ends(
            self._low_medium_lower_end,
            self._low_medium_upper_end,
            self._medium_high_lower_end,
            self._medium_high_upper_end,
        )

    def assign_energy_scale(self, value: float) -> EnergyScaleFraction:
        if value <= self._low_medium_lower_end:
            return EnergyScaleFraction(EnergyScale.LOW, 1.0, 0.0)
        elif self._low_medium_lower_end < value <= self._low_medium_upper_end:
            lower_fraction = smooth_01_transition(value, self._low_medium_lower_end, self._low_medium_upper_end)
            upper_fraction = 1.0 - lower_fraction
            return EnergyScaleFraction(EnergyScale.MIXED_LOW_MEDIUM, lower_fraction, upper_fraction)
        elif self._low_medium_upper_end < value <= self._medium_high_lower_end:
            return EnergyScaleFraction(EnergyScale.MEDIUM, 1.0, 0.0)
        elif self._medium_high_lower_end < value <= self._medium_high_upper_end:
            lower_fraction = smooth_01_transition(value, self._medium_high_lower_end, self._medium_high_upper_end)
            upper_fraction = 1.0 - lower_fraction
            return EnergyScaleFraction(EnergyScale.MIXED_MEDIUM_HIGH, lower_fraction, upper_fraction)
        else:
            return EnergyScaleFraction(EnergyScale.HIGH, 1.0, 0.0)

    def _check_width_is_nonnegative(self, width: float) -> None:
        if width < 0.0:
            raise ValueError("The width of a boundary cannot be negative.")

    def _check_order_of_ends(
        self,
        low_medium_lower_end: float,
        low_medium_upper_end: float,
        medium_high_lower_end: float,
        medium_high_upper_end: float,
    ) -> None:
        if not (low_medium_lower_end < low_medium_upper_end < medium_high_lower_end < medium_high_upper_end):
            raise ValueError(
                "The separations between the low, medium, and high ends overlap in ways that don't make sense."
            )
