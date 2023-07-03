import pytest

import numpy as np

from nn_fourbody_potential.reserved_deque import ReservedDeque


@pytest.fixture
def default_deque_246() -> ReservedDeque[int]:
    vec = ReservedDeque[int].new(3, int)
    vec.push_back(2)
    vec.push_back(4)
    vec.push_back(6)

    return vec


class TestReservedDeque:
    def test_basic(self, default_deque_246: ReservedDeque[int]):
        elements = default_deque_246.elements
        assert elements[0] == 2
        assert elements[1] == 4
        assert elements[2] == 6

    def test_classmethod_from_array(self):
        expected = ReservedDeque[int].new(3, int)
        expected.push_back(123)
        expected.push_back(456)
        expected.push_back(789)

        actual = ReservedDeque[int].from_array(np.array([123, 456, 789]))

        assert expected == actual

    def test_push_back_matches_pop_back(self, default_deque_246: ReservedDeque[int]):
        assert default_deque_246.pop_back() == 6
        assert default_deque_246.pop_back() == 4
        assert default_deque_246.pop_back() == 2

    def test_push_back_matches_pop_front(self, default_deque_246: ReservedDeque[int]):
        assert default_deque_246.pop_front() == 2
        assert default_deque_246.pop_front() == 4
        assert default_deque_246.pop_front() == 6

    def test_pop_front_then_pop_back(self, default_deque_246: ReservedDeque[int]):
        assert default_deque_246.pop_front() == 2
        assert default_deque_246.pop_back() == 6
        assert default_deque_246.pop_front() == 4

    def test_pop_back_then_pop_front(self, default_deque_246: ReservedDeque[int]):
        assert default_deque_246.pop_back() == 6
        assert default_deque_246.pop_front() == 2
        assert default_deque_246.pop_back() == 4

    def test_pop_front_then_push_front(self, default_deque_246: ReservedDeque[int]):
        assert default_deque_246.pop_front() == 2
        default_deque_246.push_front(123)
        assert default_deque_246.pop_front() == 123

    def test_raises_push_back_too_many(self, default_deque_246: ReservedDeque[int]):
        with pytest.raises(IndexError):
            default_deque_246.push_back(8)

    def test_raises_pop_back_too_many(self, default_deque_246: ReservedDeque[int]):
        default_deque_246.pop_back()
        default_deque_246.pop_back()
        default_deque_246.pop_back()

        with pytest.raises(IndexError):
            default_deque_246.pop_back()

    def test_raises_pop_front_too_many(self, default_deque_246: ReservedDeque[int]):
        default_deque_246.pop_front()
        default_deque_246.pop_front()
        default_deque_246.pop_front()

        with pytest.raises(IndexError):
            default_deque_246.pop_front()

    def test_max_size(self):
        max_size = 3
        deque = ReservedDeque[int].new(max_size, int)
        assert deque.max_size == max_size

    @pytest.mark.parametrize("max_size, elements_to_push, expected_size", [(3, [], 0), (3, [0], 1), (3, [123, 456], 2)])
    def test_size_with_push_back(self, max_size, elements_to_push, expected_size):
        deque = ReservedDeque[int].new(max_size, int)
        for elem in elements_to_push:
            deque.push_back(elem)

        assert deque.size == expected_size

    def test_size_with_push_back_and_pop_front(self):
        deque = ReservedDeque[int].new(10, int)
        deque.push_back(1)
        deque.push_back(2)
        deque.push_back(3)
        deque.pop_front()

        assert deque.size == 2

    def test_dunder_getitem(self):
        deque = ReservedDeque[int].new(10, int)
        deque.push_back(123)
        deque.push_back(456)
        deque.push_back(789)

        assert deque[0] == 123
        assert deque[1] == 456
        assert deque[2] == 789

    def test_raises_dunder_getitem_past_end(self, default_deque_246):
        with pytest.raises(IndexError):
            default_deque_246[3]

    def test_raises_dunder_getitem_past_end(self, default_deque_246):
        default_deque_246.pop_front()
        with pytest.raises(IndexError):
            default_deque_246[0]

    def test_reset(self):
        deque = ReservedDeque[float].new(10, float)
        deque.push_back(1.1)
        deque.push_back(2.2)
        deque.push_back(3.3)

        assert deque.max_size == 10
        assert deque.size == 3

        deque.reset()

        assert deque.max_size == 10
        assert deque.size == 0
