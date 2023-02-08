# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines class `InverseSquareLaw`.

"""

import numpy as np
import numpy.typing as npt

from nbpy.phasespace import PhaseSpace
from .interaction import Interaction


class InverseSquareLaw(Interaction):
    """
    Newton's classic inverse-square law of gravitation.

    Attributes
    ----------

    `constant` : float
    The gravitational constant.

    `softening` : float
    The softening parameter.

    Functions
    ---------

    `exert(self, phsp, masses)`
    Sets the accelerations according to Newton's inverse-square law.

    `name` : str
    The name of the class: "InverseSquareLaw"

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

    @classmethod
    def name(cls) -> str:
        """
        The name of the class.

        """
        return "InverseSquareLaw"

    def exert(self, phsp: PhaseSpace, masses: npt.NDArray) -> None:
        """
        Set accelerations of all interacting particles using Newton's
        inverse-square law.

        Parameters
        ----------

        `phsp` : nbpy.phasespace.PhaseSpace [mutates]
        The `PhaseSpace` object of the system. Must contain an item of key
        `Accelerations`, whose value will be set to new values by this function.

        `masses` : numpy.typing.NDArray
        N-by-1 array containing the masses of all N particles.

        """
        # `x` stores x coordinates of all particles, and similarly for y and z.
        x = phsp.positions[:, 0:1]
        y = phsp.positions[:, 1:2]
        z = phsp.positions[:, 2:3]

        # Index convention: d_x[j, k] = x[k] - x[j], and similarly for y and z.
        d_x = x.T - x
        d_y = y.T - y
        d_z = z.T - z

        inv_d_cube = (d_x**2. + d_y**2. + d_z**2. +
                      self._softening**2.)**(-1.5)

        phsp.accelerations[:, 0] = np.matmul(d_x * inv_d_cube, masses)
        phsp.accelerations[:, 1] = np.matmul(d_y * inv_d_cube, masses)
        phsp.accelerations[:, 2] = np.matmul(d_z * inv_d_cube, masses)
