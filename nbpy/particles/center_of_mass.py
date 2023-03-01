# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines functions of the phase space of the system.

"""

import numpy as np
import numpy.typing as npt

from .phase_space import PhaseSpace


def center_of_mass(phsp: PhaseSpace, masses: npt.NDArray) -> npt.NDArray:
    """
    Compute the center of mass of the system.

    Parameters
    ----------

    phsp : :class:`.PhaseSpace`
        An object containing the positions of the system.

    masses : numpy.typing.NDArray
        The masses, stored as an N-dimensional array.

    Returns
    -------

    out : numpy.typing.NDArray
        The center of mass.

    """
    return np.matmul(masses, phsp.positions) / np.sum(masses)
