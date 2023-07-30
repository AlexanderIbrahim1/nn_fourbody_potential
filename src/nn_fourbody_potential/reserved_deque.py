"""
The ReservedDeque class allows one to create an empty numpy array, and push elements
back into its 0th dimension.

This is similar to how in C++, one can create a `std::vector`, call `reserve()`, then
call `push_back()` without having to reallocate memory.
"""

from typing import Any
from typing import Generic
from typing import Iterator
from typing import Optional
from typing import Sequence
from typing import TypeVar
from typing import Union

import numpy as np


T = TypeVar("T")


# TODO: add method to "reset" the ReservedVector without having to actually clear out the elements;
#       the "effective size" of the vector is determined by `self._i_elem_start` and `self._i_elem_end`,
#       and as long as the total size remains the same, we can "reset" the vector just be setting both
#       of those values to 0


class ReservedDeque(Generic[T]):
    def __init__(
        self, elements: np.ndarray[T], i_elem_start: Optional[int] = None, i_elem_end: Optional[int] = None
    ) -> None:
        self._elements = elements

        if i_elem_start is None:
            self._i_elem_start = 0
        else:
            self._i_elem_start = i_elem_start

        if i_elem_end is None:
            self._i_elem_end = elements.shape[0]
        else:
            self._i_elem_end = i_elem_end

        if self._i_elem_start < 0:
            raise ValueError("The starting index must be 0 or greater.")

        if self._i_elem_start > elements.shape[0]:
            raise ValueError("The ending index cannot exceed the length of the underlying numpy array.")

    @classmethod
    def new(cls, shape: Union[int, Sequence[int]], type_t: T) -> "ReservedDeque[T]":
        """
        NOTE: I cannot set `dtype=T` for the numpy array because it will throw an error such as
            `TypeError: Cannot interpret '~T' as a data type`
        This is why the `type_t` variable must be redundantly passed into the constructor
        """
        i_elem_start = 0
        i_elem_end = 0
        return cls(np.empty(shape, dtype=type_t), i_elem_start, i_elem_end)

    @classmethod
    def from_array(cls, elements: np.ndarray[T]) -> "ReservedDeque[T]":
        i_elem_start = 0
        i_elem_end = len(elements)
        return cls(elements, i_elem_start, i_elem_end)

    def reset(self) -> None:
        """
        Effectively clears the data structure, allowing the user to act as if it were empty.

        The reason this works, is that all the functionality of this data structure, such as
        how it allows elements to be pushed and popped off either end, its size, and random
        access, all depend on the starting and ending indices.

        We can thus effectively "empty" the underlying data by just setting both indices to 0.
        """
        self._i_elem_start = 0
        self._i_elem_end = 0

    def push_back(self, elem: T) -> None:
        if self._i_elem_end >= self._elements.size:
            raise IndexError("Cannot push an element to the end beyond the reserved range.")

        self._elements[self._i_elem_end] = elem
        self._i_elem_end += 1

    def pop_back(self) -> T:
        if self._i_elem_end == self._i_elem_start:
            raise IndexError("Cannot pop the element off the end; there are no elements.")

        ret_value = self._elements[self._i_elem_end - 1]
        self._i_elem_end -= 1

        return ret_value

    def push_front(self, elem: T) -> None:
        if self._i_elem_start == 0:
            raise IndexError("Cannot push an element to the start beyond the reserved range.")

        self._i_elem_start -= 1
        self._elements[self._i_elem_start] = elem

    def pop_front(self) -> T:
        if self._i_elem_start == self._i_elem_end:
            raise IndexError("Cannot pop the element off the start; there are no elements.")

        ret_value = self._elements[self._i_elem_start]
        self._i_elem_start += 1

        return ret_value

    @property
    def elements(self) -> np.ndarray[T]:
        return self._elements[self._i_elem_start : self._i_elem_end]

    @property
    def max_size(self) -> int:
        return self._elements.size

    @property
    def size(self) -> int:
        return self._i_elem_end - self._i_elem_start

    def __iter__(self) -> Iterator:
        self._i_iter = self._i_elem_start
        return self

    def __next__(self) -> T:
        if self._i_iter >= self._i_elem_end:
            raise StopIteration

        value = self._elements[self._i_iter]
        self._i_iter += 1

        return value

    def __eq__(self, other: Any) -> bool:
        # NOTE: `Subscripted generics cannot be used with class and instance checks`; so I cannot use
        #       `ReservedDeque[T]` here, and I have to settle for just `ReservedDeque`
        if not isinstance(other, ReservedDeque):
            return NotImplemented

        return (
            self.max_size == other.max_size
            and self.size == other.size
            and self._i_elem_start == other._i_elem_start
            and self._i_elem_end == other._i_elem_end
            and all([self_val == other_val for (self_val, other_val) in zip(self, other)])
        )

    def __getitem__(self, index: int) -> T:
        if not self._i_elem_start <= index < self._i_elem_end:
            raise IndexError(
                "Element access is out of bounds. Must be `lower index` <= index < `upper index`\n"
                f"lower index: {self._i_elem_start}\n"
                f"upper index: {self._i_elem_end}\n"
                f"index: {index}"
            )

        return self._elements[index]