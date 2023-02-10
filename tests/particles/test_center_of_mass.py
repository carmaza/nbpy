# Distributed under the MIT License.
# See LICENSE for details.
"""
Contains unit tests for `particles.center_of_mass`.

"""

import unittest

import numpy as np

from nbpy import particles


class TestCenterOfMass(unittest.TestCase):
    """
    Test `particles.center_of_mass`.

    """

    def test(self):
        """
        Test general implementation.

        """

        seed = np.random.randint(0, 1e6)
        np.random.seed(seed)

        dim = 3
        N = np.random.randint(2, 10)
        masses = np.random.rand(N)

        phsp = particles.PhaseSpace(N)
        phsp.set_positions(np.random.randn(N, dim))

        total_mass = sum(masses)
        center_of_mass = particles.center_of_mass(phsp, masses)

        center_of_mass_expected = np.zeros(3)
        for k, mass_k in enumerate(masses):
            center_of_mass_expected += mass_k * phsp.positions[k]
        center_of_mass_expected /= total_mass

        self.assertTrue(np.allclose(center_of_mass, center_of_mass_expected),
                        msg="center_of_mass differs from expected value. "
                        f"RNG seed: {seed}.")
