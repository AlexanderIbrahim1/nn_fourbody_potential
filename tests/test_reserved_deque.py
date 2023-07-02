import pytest

from nn_fourbody_potential.full_range.extrapolated_potential import ReservedDeque


@pytest.fixture
def default_deque_246() -> ReservedDeque[int]:
    vec = ReservedDeque[int].new(3, int)
    vec.push_back(2)
    vec.push_back(4)
    vec.push_back(6)

    return vec


class TestReservedDeque:
    def test_basic(self, default_deque_246: ReservedDeque[int]):
        size = 3

        elements = default_deque_246.elements
        assert elements[0] == 2
        assert elements[1] == 4
        assert elements[2] == 6

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
