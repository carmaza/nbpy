# Distributed under the MIT License.
# See LICENSE for details.
"""
Contains unit tests for functions in module `phase_space`.

"""

from context import nbpy

import numpy as np
import unittest

import nbpy.phase_space as phase_space


class TestPhaseSpace(unittest.TestCase):
    """
    Test functions in module `phase_space`.

    """

    def test(self):
        """
        Test `center_of_mass`.

        """

        seed = np.random.randint(0, 1e6)
        np.random.seed(seed)

        dim = 3
        N = np.random.randint(2, 10)
        masses = np.random.rand(N)
        positions = np.random.randn(N, dim)

        total_mass = sum(masses)
        center_of_mass = phase_space.center_of_mass(masses, positions)

        center_of_mass_expected = np.zeros(3)
        for k, mass_k in enumerate(masses):
            center_of_mass_expected += mass_k * positions[k]
        center_of_mass_expected /= total_mass

        self.assertTrue(np.allclose(center_of_mass, center_of_mass_expected),
                        msg="center_of_mass differs from expected value. "
                        f"RNG seed: {seed}.")


if __name__ == "__main__":
    unittest.main()
