# Distributed under the MIT License.
# See LICENSE for details.

import numpy as np


class RandomDistribution:
    """
    Set positions and velocities to random values normally distributed.
    The normal distribution is fed the given `seed`.

    """

    def __init__(self, seed=25092020):
        self._seed = seed

    @property
    def seed(self):
        return self._seed

    def set_variables(self, positions, velocities):
        N = positions.shape[0]
        rng = np.random.default_rng(self._seed)
        positions[:] = rng.standard_normal((N, 3))
        velocities[:] = rng.standard_normal((N, 3))
