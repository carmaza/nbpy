# Distributed under the MIT License.
# See LICENSE for details.

from context import nbpy

import numpy as np
import unittest

from nbpy.inverse_square_law import InverseSquareLaw
from nbpy.leapfrog import Leapfrog


class TestLeapfrog(unittest.TestCase):
    """
    Test functions in `Leapfrog` class.
    """

    @staticmethod
    def name():
        return "TestLeapfrog"

    def test(self):

        seed = np.random.randint(0, 1e6)
        np.random.seed(seed)

        dim = 3
        N = np.random.randint(2, 10)
        masses = np.random.rand(N)

        # The specific law we use is unimportant.
        law = InverseSquareLaw(np.random.rand(), np.random.rand())

        pos = np.random.randn(N, dim)
        vel = np.random.randn(N, dim)
        acc = np.empty_like(pos)
        law.exert(acc, masses, pos)

        # Copies to compare with later.
        pos_ini = pos.copy()
        vel_ini = vel.copy()
        acc_ini = acc.copy()

        # Update evolved variables.
        dt = np.random.randn()
        stepper = Leapfrog()
        stepper.evolve(pos, vel, acc, dt, masses, law)

        # Construct expected evolved variables.
        v_halfstep = vel_ini + 0.5 * dt * acc_ini
        pos_expected = pos_ini + dt * v_halfstep

        acc_fin = np.empty_like(pos_expected)
        law.exert(acc_fin, masses, pos_expected)
        vel_expected = v_halfstep + 0.5 * dt * acc_fin

        self.assertTrue(
            np.allclose(pos, pos_expected),
            msg="In {name}: new position differs from expected value. "
            "RNG seed: {seed}.".format(name=self.name(), seed=seed))

        self.assertTrue(
            np.allclose(vel, vel_expected),
            msg="In {name}: new velocity differs from expected value. "
            "RNG seed: {seed}.".format(name=self.name(), seed=seed))

        self.assertTrue(
            np.allclose(acc, acc_fin),
            msg="In {name}: new acceleration differs from expected value. "
            "RNG seed: {seed}.".format(name=self.name(), seed=seed))

        print("\nAll tests in {s} passed.".format(s=self.name()))


if __name__ == "__main__":
    unittest.main()
