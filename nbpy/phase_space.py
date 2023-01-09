# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines the following functions of the phase space:
- center_of_mass(masses, positions)
"""

import numpy as np


def center_of_mass(masses, positions):
    """
    Compute the center of mass of the system.

    Parameters
    ----------

    `masses` : ndarray
    The masses, stored as an N-dimensional NumPy array.

    `positions` : ndarray
    The 3-d positions of the particles, stored as N-by-3 NumPy arrays.

    Returns
    -------

    out : ndarray
    The center of mass.

    """
    return np.matmul(masses, positions) / np.sum(masses)
