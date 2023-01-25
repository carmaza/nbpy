# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines class `RandomDistribution`.

"""

import numpy as np
import numpy.typing as npt


class RandomDistribution:
    """
    Sets positions and velocities to random values normally distributed.

    Attributes
    ----------

    seed : int (default: 25092020)
    The RNG seed.

    Functions
    ---------

    `set_variables(positions, velocities)`
    Assigns positions and velocities to random numbers.

    """

    def __init__(self, seed: int = 25092020):
        """
        Parameters
        ----------

        `seed` : int (default: 25092020)
        The RNG seed.

        """
        self._seed = seed

    @property
    def seed(self) -> int:
        """
        The RNG seed used to generate the distribution.

        """
        return self._seed

    def set_variables(self, positions: npt.NDArray,
                      velocities: npt.NDArray) -> None:
        """
        Assign `positions` and `velocities` to random numbers.

        Parameters
        ----------

        `positions, velocities` : numpy.typing.NDArray, numpy.typing.NDArray [mutate]
        The N positions and velocities, represented as N-by-3 arrays.

        """
        N = positions.shape[0]
        rng = np.random.default_rng(self._seed)
        positions[:] = rng.standard_normal((N, 3))
        velocities[:] = rng.standard_normal((N, 3))
