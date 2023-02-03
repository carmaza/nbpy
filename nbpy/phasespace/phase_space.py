# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines class `phasespace.PhaseSpace`.

"""

import numpy as np
import numpy.typing as npt


class PhaseSpace:
    """
    An associative container that holds variables related to the phase space
    of an N-body simulation. Specifically, it holds:

    - Positions: N-by-3 array.
    - Velocities: N-by-3 array.
    - Accelerations: N-by-3 array.

    Attributes
    ----------

    - `positions`: numpy.typing.NDArray
      The positions of the particles.

    - `velocities`: numpy.typing.NDArray
      The velocities of the particles

    - `velocities`: numpy.typing.NDArray
      The positions of the particles

    Functions
    ---------

    - `set_positions(value)`
      Sets positions to new value `value`.

    - `set_velocities(value)`
      Sets velocities to new value `value`.

    - `set_accelerations(value)`
      Sets accelerations to new value `value`.

    Notes
    -----

    - All variables are first initialized to NaNs during the construction of the
      object. It is expected that the attributes are subsequently set to
      acceptable values using the provided setters.

    """

    def __init__(self, N: int):
        self._data = {}
        self._data["Positions"] = np.full((N, 3), np.nan)
        self._data["Velocities"] = np.full((N, 3), np.nan)
        self._data["Accelerations"] = np.full((N, 3), np.nan)

    @property
    def positions(self):
        """
        The positions of every particle in the system.

        """
        return self._data["Positions"]

    @property
    def velocities(self):
        """
        The velocities of every particle in the system.

        """
        return self._data["Velocities"]

    @property
    def accelerations(self):
        """
        The accelerations of every particle in the system.

        """
        return self._data["Accelerations"]

    def _set(self, key: str, value: npt.NDArray) -> None:
        self._data[key][:] = value

    def set_positions(self, value: npt.NDArray) -> None:
        """
        Set positions of every particle to the given value.

        """
        self._set("Positions", value)

    def set_velocities(self, value: npt.NDArray) -> None:
        """
        Set velocities of every particle to the given value.

        """
        self._set("Velocities", value)

    def set_accelerations(self, value: npt.NDArray) -> None:
        """
        Set accelerations of every particle to the given value.

        """
        self._set("Accelerations", value)
