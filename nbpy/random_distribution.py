# Distributed under the MIT License.
# See LICENSE for details.

import numpy as np


class RandomDistribution:
    """
    Set positions and velocities to random values normally distributed.
    The normal distribution is fed the given `seed`.

    """

    def __init__(self, seed=np.random.randint(1, 1e6)):
        self._seed = seed

    @property
    def seed(self):
        return self._seed

    def variables(self, N):
        np.random.seed(self._seed)
        pos = np.random.randn(N, 3)
        vel = np.random.randn(N, 3)
        return pos, vel
