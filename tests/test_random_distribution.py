# Distributed under the MIT License.
# See LICENSE for details.

from context import nbpy

import numpy as np
import unittest

from nbpy.random_distribution import RandomDistribution


class TestRandomDistribution(unittest.TestCase):
    """
    Test functions in `RandomDistribution` class.
    """

    @staticmethod
    def name():
        return "TestRandomDistribution"

    def test(self):

        seed = np.random.randint(0, 1e6)
        np.random.seed(seed)

        N = np.random.randint(2, 10)
        positions, velocities = RandomDistribution(seed).variables(N)

        # Reset rng to same seed in order to obtain same random arrays.
        np.random.seed(seed)

        self.assertTrue(
            np.allclose(positions, np.random.randn(N, 3)),
            msg="In {name}: positions differs from expected value. "
            "RNG seed: {seed}.".format(name=self.name(), seed=seed))

        self.assertTrue(
            np.allclose(velocities, np.random.randn(N, 3)),
            msg="In {name}: gravitational energy differs from expected value. "
            "RNG seed: {seed}.".format(name=self.name(), seed=seed))

        print("\nAll tests in {s} passed.".format(s=self.name()))


if __name__ == "__main__":
    unittest.main()
