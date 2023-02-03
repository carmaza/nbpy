# Distributed under the MIT License.
# See LICENSE for details.
"""
Contains unit tests for class `phasespace.PhaseSpace`.

"""

import unittest

import numpy as np

from nbpy.phasespace import PhaseSpace


class TestPhaseSpace(unittest.TestCase):
    """
    Test class `phasespace.PhaseSpace`.

    """

    @classmethod
    def setUpClass(cls):
        cls._seed = np.random.randint(0, 1e6)
        np.random.seed(cls._seed)

        cls._N = np.random.randint(2, 10)
        cls._ps = PhaseSpace(cls._N)

    def test_attribute_interface(self):
        """
        Test interface for class attributes, including setters and getters.

        """
        dim = 3

        positions = np.random.randn(self._N, dim)
        self._ps.set_positions(positions)
        self.assertTrue(np.allclose(positions, self._ps.positions),
                        msg="positions differs from expected value. "
                        f"RNG seed: {self._seed}.")

        velocities = np.random.randn(self._N, dim)
        self._ps.set_velocities(velocities)
        self.assertTrue(np.allclose(velocities, self._ps.velocities),
                        msg="velocities differs from expected value. "
                        f"RNG seed: {self._seed}.")

        accelerations = np.random.randn(self._N, dim)
        self._ps.set_accelerations(accelerations)
        self.assertTrue(np.allclose(accelerations, self._ps.accelerations),
                        msg="accelerations differs from expected value. "
                        f"RNG seed: {self._seed}.")
