# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines class :class:`.RandomDistribution`.

"""

import numpy as np

from nbpy.particles import PhaseSpace


class RandomDistribution:
    """
    Sets positions and velocities to random values normally distributed.

    Parameters
    ----------

    seed : int (default: 25092020)
        The RNG seed.

    """

    def __init__(self, seed: int = 25092020):
        self._seed = seed

    @property
    def seed(self) -> int:
        """
        The RNG seed used to generate the distribution.

        """
        return self._seed

    def set_variables(self, phsp: PhaseSpace) -> None:
        """
        Assign positions and velocities to random numbers.

        Parameters
        ----------

        phsp : :class:`.PhaseSpace`
            The phase space of the system. Must contain items of keys
            ``"Positions"`` and ``"Velocities"``, whose values will be set by
            this function.

        """
        N = phsp.positions.shape[0]
        rng = np.random.default_rng(self._seed)
        phsp.set_positions(rng.standard_normal((N, 3)))
        phsp.set_velocities(rng.standard_normal((N, 3)))
