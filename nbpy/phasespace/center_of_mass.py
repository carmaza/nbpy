# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines the following function of the phase space:

- `center_of_mass(masses, positions)`
  Computes the center of mass of the system.

"""

import numpy as np
import numpy.typing as npt

from .phase_space import PhaseSpace


def center_of_mass(masses: npt.NDArray, phsp: PhaseSpace) -> npt.NDArray:
    """
    Compute the center of mass of the system.

    Parameters
    ----------

    `masses` : numpy.typing.NDArray
    The masses, stored as an N-dimensional array.

    `positions` : np.typing.NDarray
    The 3-d positions of the particles, stored as an N-by-3 array.

    Returns
    -------

    out : numpy.typing.NDArray
    The center of mass.

    """
    return np.matmul(masses, phsp.positions) / np.sum(masses)
