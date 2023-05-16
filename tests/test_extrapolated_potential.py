import pytest

from nn_fourbody_potential.full_range.extrapolated_potential import ReservedVector


class TestReservedVector:
    def test_basic(self):
        size = 3
        
        vec = ReservedVector[int](3, int)
        vec.push_back(2)
        vec.push_back(4)
        vec.push_back(6)
        
        elements = vec.elements
        assert elements[0] == 2
        assert elements[1] == 4
        assert elements[2] == 6
        
    def test_raises_too_many_elements(self):
        size = 3
        
        vec = ReservedVector[int](3, int)
        vec.push_back(2)
        vec.push_back(4)
        vec.push_back(6)
        
        with pytest.raises(IndexError):
            vec.push_back(8)