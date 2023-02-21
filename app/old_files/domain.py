"""
Get data from the feature set domain.
"""

from __future__ import annotations

import itertools
from typing import Optional
from typing import Tuple

import numpy as np


class DomainND:
    def __init__(self, domain_bounds: list[Tuple[float, float]]) -> None:

        for db in domain_bounds:
            assert db[0] < db[1]

        self._domain_bounds = domain_bounds

    def linear_grid(self, n_points_per_dim: list[int]) -> np.ndarray:
        assert len(n_points_per_dim) == len(self._domain_bounds)

        linspaces = []
        for (db, n_points) in zip(self._domain_bounds, n_points_per_dim):
            lower_bound = db[0]
            upper_bound = db[1]
            linspaces.append(np.linspace(lower_bound, upper_bound, n_points))

        return np.array(list(itertools.product(*linspaces)))

    def sample(self, n_samples: int) -> np.ndarray:
        assert n_samples > 0

        domain_values = np.array([
            np.random.uniform(db[0], db[1], n_samples)
            for db in self._domain_bounds
        ])

        domain_values = np.transpose(domain_values)

        return domain_values

        
