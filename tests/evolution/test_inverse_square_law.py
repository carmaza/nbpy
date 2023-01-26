# Distributed under the MIT License.
# See LICENSE for details.
"""
Contains unit tests for class `evolution.InverseSquareLaw`.

"""

import unittest

import numpy as np

from nbpy.evolution import InverseSquareLaw


class TestInverseSquareLaw(unittest.TestCase):
    """
    Test class `evolution.InverseSquareLaw`.

    """

    @classmethod
    def setUpClass(cls):
        cls._seed = np.random.randint(0, 1e6)
        np.random.seed(cls._seed)

        cls._constant = np.random.rand()
        cls._softening = np.random.rand()
        cls._law = InverseSquareLaw(cls._constant, cls._softening)

    def test_attribute_interface(self):
        """
        Test interface for class attributes.

        """
        self.assertAlmostEqual(
            self._constant,
            self._law.constant,
            msg="value of constant differs from expected value. "
            f"RNG seed: {self._seed}.")

        self.assertAlmostEqual(
            self._softening,
            self._law.softening,
            msg="value of softening differs from expected value. "
            f"RNG seed: {self._seed}.")

    def test_exert(self):
        """
        Test implementation of member function `exert`.

        """
        dim = 3
        n_body = np.random.randint(2, 10)
        positions = np.random.randn(n_body, dim)
        masses = np.random.rand(n_body)

        accelerations = np.empty_like(positions)
        self._law.exert(accelerations, masses, positions)

        accelerations_expected = np.zeros_like(positions)

        for j in range(n_body):
            for k, mass in enumerate(masses):
                d_x = positions[k, 0] - positions[j, 0]
                d_y = positions[k, 1] - positions[j, 1]
                d_z = positions[k, 2] - positions[j, 2]
                d_cube = (d_x**2. + d_y**2. + d_z**2. +
                          self._softening**2.)**1.5
                accelerations_expected[j, 0] += mass * d_x / d_cube
                accelerations_expected[j, 1] += mass * d_y / d_cube
                accelerations_expected[j, 2] += mass * d_z / d_cube

        self.assertTrue(np.allclose(accelerations, accelerations_expected),
                        msg="acceleration differs from expected value. "
                        f"RNG seed: {self._seed}.")
