"""
The ReservedVector class allows one to create an empty numpy array, and push elements
back into its 0th dimension.

This is similar to how in C++, one can create a `std::vector`, call `reserve()`, then
call `push_back()` without having to reallocate memory.
"""

from typing import Generic
from typing import Sequence
from typing import TypeVar
from typing import Union

import numpy as np


T = TypeVar("T")


class ReservedVector(Generic[T]):
    def __init__(self, shape: Union[int, Sequence[int]], type_t: T) -> None:
        """
        NOTE: I cannot set `dtype=T` for the numpy array because it will throw an error such as
            `TypeError: Cannot interpret '~T' as a data type`
        This is why the `type_t` variable must be redundantly passed into the constructor

        """
        self._elements = np.empty(shape, dtype=type_t)
        self._i_elem = 0

    def push_back(self, elem: T) -> None:
        if self._i_elem >= self._elements.size:
            raise IndexError("Cannot push back an element beyond the reserved range.")

        self._elements[self._i_elem] = elem
        self._i_elem += 1

    @property
    def elements(self) -> np.ndarray[T]:
        return self._elements

    @property
    def size(self) -> int:
        return self._elements.size

    def __getitem__(self, index: int) -> T:
        return self._elements[index]
