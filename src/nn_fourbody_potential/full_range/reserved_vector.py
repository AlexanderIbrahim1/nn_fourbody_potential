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
    def __init__(self, elements: np.ndarray[T], i_elem_start: int, i_elem_end: int) -> None:
        """
        NOTE: I cannot set `dtype=T` for the numpy array because it will throw an error such as
            `TypeError: Cannot interpret '~T' as a data type`
        This is why the `type_t` variable must be redundantly passed into the constructor

        """
        self._elements = elements
        self._i_elem_start = i_elem_start
        self._i_elem_end = i_elem_end
    
    @classmethod
    def new(cls, shape: Union[int, Sequence[int]], type_t: T) -> 'ReservedVector[T]':
        """
        NOTE: I cannot set `dtype=T` for the numpy array because it will throw an error such as
            `TypeError: Cannot interpret '~T' as a data type`
        This is why the `type_t` variable must be redundantly passed into the constructor

        """
        i_elem_start = 0
        i_elem_end = 0
        return cls(np.empty(shape, dtype=type_t), i_elem_start, i_elem_end)
    
    @classmethod
    def from_array(cls, elements: np.ndarray[T]) -> 'ReservedVector[T]':
        i_elem_start = 0
        i_elem_end = len(elements)
        return cls(elements, i_elem_start, i_elem_end)

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
        return self._elements

    @property
    def size(self) -> int:
        return self._elements.size

    def __getitem__(self, index: int) -> T:
        return self._elements[index]