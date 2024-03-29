# Distributed under the MIT License.
# See LICENSE for details.
"""
Contains unit tests for class `interactions.InverseSquareLaw`.

"""

import unittest

import numpy as np

from nbpy.interactions import InverseSquareLaw
from nbpy.particles import PhaseSpace


class TestInverseSquareLaw(unittest.TestCase):
    """
    Test class `interactions.InverseSquareLaw`.

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
        self.assertEqual("InverseSquareLaw",
                         self._law.name(),
                         msg="value of name differs from expected value. "
                         f"RNG seed: {self._seed}.")

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

    def test_construct_from_dict(self):
        """
        Test construction from dictionary.

        """
        params = {"Constant": self._constant, "Softening": self._softening}
        law_from_dict = InverseSquareLaw.from_dict(params)

        self.assertEqual(law_from_dict.name(),
                         self._law.name(),
                         msg="value of name differs from expected value. "
                         f"RNG seed: {self._seed}.")

        self.assertAlmostEqual(
            law_from_dict.constant,
            self._law.constant,
            msg="value of constant differs from expected value. "
            f"RNG seed: {self._seed}.")

        self.assertAlmostEqual(
            law_from_dict.softening,
            self._law.softening,
            msg="value of softening differs from expected value. "
            f"RNG seed: {self._seed}.")

    @unittest.expectedFailure
    def test_construct_from_dict_failure(self):
        """
        Test failed construction from dictionary.

        """
        params = {"WrongKey": self._constant, "Softening": self._softening}
        InverseSquareLaw.from_dict(params)

    def test_exert(self):
        """
        Test implementation of member function `exert`.

        """
        dim = 3
        n_body = np.random.randint(2, 10)
        masses = np.random.rand(n_body)

        phsp = PhaseSpace(n_body)
        phsp.set_positions(np.random.randn(n_body, dim))
        self._law.exert(phsp, masses)

        accelerations_expected = np.zeros_like(phsp.positions)

        for j in range(n_body):
            for k, mass in enumerate(masses):
                d_x = phsp.positions[k, 0] - phsp.positions[j, 0]
                d_y = phsp.positions[k, 1] - phsp.positions[j, 1]
                d_z = phsp.positions[k, 2] - phsp.positions[j, 2]
                d_cube = (d_x**2. + d_y**2. + d_z**2. +
                          self._softening**2.)**1.5
                accelerations_expected[j, 0] += mass * d_x / d_cube
                accelerations_expected[j, 1] += mass * d_y / d_cube
                accelerations_expected[j, 2] += mass * d_z / d_cube

        self.assertTrue(np.allclose(phsp.accelerations,
                                    accelerations_expected),
                        msg="acceleration differs from expected value. "
                        f"RNG seed: {self._seed}.")
