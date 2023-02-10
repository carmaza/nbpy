# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines the following function of the phase space:

- `center_of_mass(phsp, masses)`
  Computes the center of mass of the system.

"""

import numpy as np
import numpy.typing as npt

from .phase_space import PhaseSpace


def center_of_mass(phsp: PhaseSpace, masses: npt.NDArray) -> npt.NDArray:
    """
    Compute the center of mass of the system.

    Parameters
    ----------

    `phsp` : nbpy.particles.PhaseSpace
    A `PhaseSpace` object containing the positions of the system. Must contain
    an item of key "Positions".

    `masses` : numpy.typing.NDArray
    The masses, stored as an N-dimensional array.

    Returns
    -------

    out : numpy.typing.NDArray
    The center of mass.

    """
    return np.matmul(masses, phsp.positions) / np.sum(masses)
