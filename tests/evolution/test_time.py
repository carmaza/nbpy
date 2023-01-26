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

    def test(self):
        """
        Test class and member functions.

        """

        seed = np.random.randint(0, 1e6)
        np.random.seed(seed)

        id_ = np.random.randint(100)
        value = np.random.randn()
        time = Time(id_, value)

        self.assertEqual(time.id_,
                         id_,
                         msg="time ID differs from expected value. "
                         f"RNG seed: {seed}.")

        self.assertAlmostEqual(time.value,
                               value,
                               msg="time value differs from expected value. "
                               f"RNG seed: {seed}.")
