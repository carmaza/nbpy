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

        N = np.random.randint(2, 10)
        positions = np.zeros((N, 3))
        velocities = np.zeros((N, 3))
        RandomDistribution(seed).set_variables(positions, velocities)

        # Reset RNG to previous seed in order to obtain same random arrays.
        rng = np.random.default_rng(seed)

        self.assertTrue(
            np.allclose(positions, rng.standard_normal((N, 3))),
            msg="In {name}: positions differs from expected value. "
            "RNG seed: {seed}.".format(name=self.name(), seed=seed))

        self.assertTrue(
            np.allclose(velocities, rng.standard_normal((N, 3))),
            msg="In {name}: gravitational energy differs from expected value. "
            "RNG seed: {seed}.".format(name=self.name(), seed=seed))

        print("\nAll tests in {s} passed.".format(s=self.name()))


if __name__ == "__main__":
    unittest.main()
