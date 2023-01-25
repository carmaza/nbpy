# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines class `InverseSquareLaw`.

"""

import numpy as np
import numpy.typing as npt


class InverseSquareLaw:
    """
    Newton's classic inverse square law.

    Attributes
    ----------

    `constant` : float
    The gravitational constant.

    `softening` : float
    The softening parameter.

    Functions
    ---------

    `exert(self, accelerations, masses, positions)`
    Calculates the accelerations from the given parameters.

    """

    def __init__(self, constant: float, softening: float):
        """
        Parameters
        ----------

        `constant` : float
        The gravitational constant.

        `softening` : float
        the softening parameter.

        """
        self._constant = constant
        self._softening = softening

    @property
    def constant(self) -> float:
        """
        The gravitational constant.

        """
        return self._constant

    @property
    def softening(self) -> float:
        """
        The softening parameter for close encounters.

        """
        return self._softening

    def exert(self, accelerations: npt.NDArray, masses: npt.NDArray,
              positions: npt.NDArray) -> None:
        """
        Compute accelerations of all interacting particles.

        Parameters
        ----------

        `accelerations` : numpy.typing.NDArray [mutates]
        N-by-3 array to store the accelerations of all N particles.

        `masses` : numpy.typing.NDArray
        N-by-1 array containing the masses of all N particles.

        `positions` : numpy.typing.NDArray
        N-by-3 array containing the positions of all N particles.

        """
        # `x` stores x coordinates of all particles, and similarly for y and z.
        x = positions[:, 0:1]
        y = positions[:, 1:2]
        z = positions[:, 2:3]

        # Index convention: d_x[j, k] = x[k] - x[j], and similarly for y and z.
        d_x = x.T - x
        d_y = y.T - y
        d_z = z.T - z

        inv_d_cube = (d_x**2. + d_y**2. + d_z**2. +
                      self._softening**2.)**(-1.5)

        accelerations[:, 0] = np.matmul(d_x * inv_d_cube, masses)
        accelerations[:, 1] = np.matmul(d_y * inv_d_cube, masses)
        accelerations[:, 2] = np.matmul(d_z * inv_d_cube, masses)
