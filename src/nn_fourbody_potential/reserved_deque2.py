"""
The ReservedDeque class allows one to create an empty numpy array, and push elements
back into its 0th dimension.

This is similar to how in C++, one can create a `std::vector`, call `reserve()`, then
call `push_back()` without having to reallocate memory.
"""

from typing import Any
from typing import Iterator
from typing import Sequence


class ReservedDeque:
    def __init__(self, elements: Sequence[Any], i_elem_start: int, i_elem_end: int) -> None:
        self._elements = elements
        self._max_size = len(self._elements)
        self._i_elem_start = i_elem_start
        self._i_elem_end = i_elem_end

    @classmethod
    def with_no_size(cls, elements: Sequence[Any]) -> None:
        return cls(elements, 0, 0)

    @classmethod
    def with_size(cls, elements: Sequence[Any]) -> None:
        return cls(elements, 0, len(elements))

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

    def push_back(self, elem: Any) -> None:
        if self._i_elem_end >= self._max_size:
            raise IndexError("Cannot push an element to the end beyond the reserved range.")

        self._elements[self._i_elem_end] = elem
        self._i_elem_end += 1

    def pop_back(self) -> Any:
        if self._i_elem_end == self._i_elem_start:
            raise IndexError("Cannot pop the element off the end; there are no elements.")

        ret_value = self._elements[self._i_elem_end - 1]
        self._i_elem_end -= 1

        return ret_value

    def push_front(self, elem: Any) -> None:
        if self._i_elem_start == 0:
            raise IndexError("Cannot push an element to the start beyond the reserved range.")

        self._i_elem_start -= 1
        self._elements[self._i_elem_start] = elem

    def pop_front(self) -> Any:
        if self._i_elem_start == self._i_elem_end:
            raise IndexError("Cannot pop the element off the start; there are no elements.")

        ret_value = self._elements[self._i_elem_start]
        self._i_elem_start += 1

        return ret_value

    @property
    def elements(self) -> Sequence[Any]:
        return self._elements[self._i_elem_start : self._i_elem_end]

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def size(self) -> int:
        return self._i_elem_end - self._i_elem_start

    def __iter__(self) -> Iterator:
        self._i_iter = self._i_elem_start
        return self

    def __next__(self) -> Any:
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
            self._max_size == other._max_size
            and self._i_elem_start == other._i_elem_start
            and self._i_elem_end == other._i_elem_end
            and all([self_val == other_val for (self_val, other_val) in zip(self, other)])
        )

    def __getitem__(self, index: int) -> Any:
        if not self._i_elem_start <= index < self._i_elem_end:
            raise IndexError(
                "Element access is out of bounds. Must be `lower index` <= index < `upper index`\n"
                f"lower index: {self._i_elem_start}\n"
                f"upper index: {self._i_elem_end}\n"
                f"index: {index}"
            )

        return self._elements[index]
