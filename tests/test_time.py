# Distributed under the MIT License.
# See LICENSE for details.
"""
Contains unit tests for class `Time`.

"""

from context import nbpy

import numpy as np
import unittest

from nbpy.time import Time


class TestTime(unittest.TestCase):
    """
    Test `Time` class.
    """

    def test(self):

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


if __name__ == "__main__":
    unittest.main()
