# Distributed under the MIT License.
# See LICENSE for details.
"""
Contains unit tests for class `RandomDistribution`.

"""

from context import nbpy

import numpy as np
import unittest

from nbpy.random_distribution import RandomDistribution


class TestRandomDistribution(unittest.TestCase):
    """
    Test functions in `RandomDistribution` class.
    """

    def test(self):

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


if __name__ == "__main__":
    unittest.main()
