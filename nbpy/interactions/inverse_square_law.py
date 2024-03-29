# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines class :class:`.InverseSquareLaw`.

"""
import numpy as np
import numpy.typing as npt

from nbpy.particles import PhaseSpace
from .interaction import Interaction


class InverseSquareLaw(Interaction):
    """
    Newton's classic inverse-square law of gravitation.

    Parameters
    ----------

    constant : float
        The gravitational constant.

    softening : float
        The softening parameter.

    """

    def __init__(self, constant: float, softening: float):
        self._constant = constant
        self._softening = softening

    @classmethod
    def name(cls) -> str:
        """
        The name of the class.

        """
        return cls.__name__

    @classmethod
    def from_dict(cls, params: dict) -> 'InverseSquareLaw':
        """
        Construct an instance from a dictionary of parameters.

        Parameters
        ----------

        params : dict
            The dictionary. Must contain keys ``"Constant"`` (float),
            and ``"Softening"`` (float).

        Returns
        -------

        out : obj
            The constructed object.

        """
        try:
            return cls(params["Constant"], params["Softening"])
        except KeyError:
            print(f"""KeyError in {cls.name()}:\n:
            Keys expected: ['Constant', 'Softening'].
            Keys passed:   {list(params.keys())}
            """)
            raise
        except:
            print("Unknown error when initializing {cls.name()} from dict.")
            raise

    @property
    def constant(self) -> float:
        """
        The gravitational constant.

        """
        return self._constant

    @property
    def softening(self) -> float:
        """
        The softening parameter.

        """
        return self._softening

    def exert(self, phsp: PhaseSpace, masses: npt.NDArray) -> None:
        """
        Set accelerations of all interacting particles using Newton's
        inverse-square law.

        Parameters
        ----------

        phsp : :class:`.PhaseSpace`
            The object representing the system's phase space. Must contain a key
            ``"Accelerations"``, whose value will be set to new values by this
            function.

        masses : numpy.typing.NDArray
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
