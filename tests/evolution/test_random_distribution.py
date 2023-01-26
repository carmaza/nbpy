# Distributed under the MIT License.
# See LICENSE for details.
"""
Contains unit tests for class `RandomDistribution`.

"""

import unittest

import numpy as np

from nbpy.evolution import RandomDistribution


class TestRandomDistribution(unittest.TestCase):
    """
    Test class `RandomDistribution`.

    """

    def test(self):
        """
        Test class and member functions.

        """

        seed = np.random.randint(0, 1e6)

        N = np.random.randint(2, 10)
        positions = np.zeros((N, 3))
        velocities = np.zeros((N, 3))
        RandomDistribution(seed).set_variables(positions, velocities)

        # Reset RNG to previous seed in order to obtain same random arrays.
        rng = np.random.default_rng(seed)

        self.assertTrue(np.allclose(positions, rng.standard_normal((N, 3))),
                        msg="positions differs from expected value. "
                        f"RNG seed: {seed}.")

        self.assertTrue(
            np.allclose(velocities, rng.standard_normal((N, 3))),
            msg="gravitational energy differs from expected value. "
            f"RNG seed: {seed}.")