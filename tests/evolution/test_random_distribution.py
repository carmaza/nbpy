# Distributed under the MIT License.
# See LICENSE for details.
"""
Contains unit tests for class `evolution.RandomDistribution`.

"""

import unittest

import numpy as np

from nbpy.evolution import RandomDistribution


class TestRandomDistribution(unittest.TestCase):
    """
    Test class `evolution.RandomDistribution`.

    """

    @classmethod
    def setUpClass(cls):
        cls._seed = np.random.randint(0, 1e6)
        np.random.seed(cls._seed)

        # The member `seed` coincides with the seed used for testing using RNGs.
        cls._dist = RandomDistribution(cls._seed)

    def test_attribute_interface(self):
        """
        Test interface for class attributes.

        """
        self.assertEqual(self._seed,
                         self._dist.seed,
                         msg="value of seed differs from expected value."
                         f"RNG seed: {self._seed}.")

    def test_set_variables(self):
        """
        Test implementation of member function `set_variables`.

        """

        N = np.random.randint(2, 10)
        positions = np.zeros((N, 3))
        velocities = np.zeros((N, 3))
        self._dist.set_variables(positions, velocities)

        # Reset RNG to previous seed in order to obtain same random arrays.
        rng = np.random.default_rng(self._seed)

        self.assertTrue(np.allclose(positions, rng.standard_normal((N, 3))),
                        msg="positions differs from expected value. "
                        f"RNG seed: {self._seed}.")

        self.assertTrue(
            np.allclose(velocities, rng.standard_normal((N, 3))),
            msg="gravitational energy differs from expected value. "
            f"RNG seed: {self._seed}.")
