# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines the `RandomDistribution` class.
"""

import numpy as np


class RandomDistribution:
    """
    Sets positions and velocities to random values normally distributed.

    Attributes
    ----------
    seed : int (default: 25092020)
    The RNG seed.

    Functions
    ---------
    set_variables(positions, velocities)
    Assigns positions and velocities to random numbers.
    """

    def __init__(self, seed=25092020):
        self._seed = seed

    @property
    def seed(self):
        """
        The RNG seed used to generate the distribution.
        """
        return self._seed

    def set_variables(self, positions, velocities):
        """
        Assign `positions` and `velocities` to random numbers.

        Parameters
        ----------
        `positions, velocities` : ndarray, ndarray [mutate]
        The N positions and velocities, represented as N-by-3 numpy arrays.

        """
        N = positions.shape[0]
        rng = np.random.default_rng(self._seed)
        positions[:] = rng.standard_normal((N, 3))
        velocities[:] = rng.standard_normal((N, 3))
