# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines the following function of the phase space:

- `center_of_mass(masses, phsp)`
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

    `phsp` : nbpy.phasespace.PhaseSpace
    A `PhaseSpace` object containing the positions of the system.

    Returns
    -------

    out : numpy.typing.NDArray
    The center of mass.

    """
    return np.matmul(masses, phsp.positions) / np.sum(masses)
