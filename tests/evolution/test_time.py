# Distributed under the MIT License.
# See LICENSE for details.
"""
Contains unit tests for class `evolution.Time`.

"""

import unittest

import numpy as np

from nbpy.evolution import Time


class TestTime(unittest.TestCase):
    """
    Test class `evolution.Time`.

    """

    @classmethod
    def setUpClass(cls):
        cls._seed = np.random.randint(0, 1e6)
        np.random.seed(cls._seed)

        cls._id_ = np.random.randint(100)
        cls._value = np.random.randn()
        cls._time = Time(cls._id_, cls._value)

    def test_attribute_interface(self):
        """
        Test interface for class attributes.

        """
        self.assertEqual(self._time.id_,
                         self._id_,
                         msg="time ID differs from expected value. "
                         f"RNG seed: {self._seed}.")

        self.assertAlmostEqual(self._time.value,
                               self._value,
                               msg="time value differs from expected value. "
                               f"RNG seed: {self._seed}.")
