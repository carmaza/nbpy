# Distributed under the MIT License.
# See LICENSE for details.
"""
Contains unit tests for class `evolution.Leapfrog`.

"""
import unittest

import numpy as np

from nbpy.evolution import InverseSquareLaw
from nbpy.evolution import Leapfrog
from nbpy.phasespace import PhaseSpace


class TestLeapfrog(unittest.TestCase):
    """
    Test class `evolution.Leapfrog`.

    """

    @classmethod
    def setUpClass(cls):
        cls._seed = np.random.randint(0, 1e6)
        np.random.seed(cls._seed)

        cls._stepper = Leapfrog()

        # The specific law we use is unimportant.
        cls._law = InverseSquareLaw(np.random.rand(), np.random.rand())

    def test_evolve(self):
        """
        Test implementation of member function `evolve`.

        """
        dim = 3
        N = np.random.randint(2, 10)
        masses = np.random.rand(N)

        phsp = PhaseSpace(N)
        phsp.set_positions(np.random.randn(N, dim))
        phsp.set_velocities(np.random.randn(N, dim))
        self._law.exert(phsp, masses)

        # New phase space to compare with later.
        phsp_expected = PhaseSpace(N)
        phsp_expected.set_positions(phsp.positions)
        phsp_expected.set_velocities(phsp.velocities)
        phsp_expected.set_accelerations(phsp.accelerations)

        # Update evolved variables.
        dt = np.random.randn()
        self._stepper.evolve(phsp, dt, masses, self._law)

        # Construct expected evolved variables.
        v_halfstep = phsp_expected.velocities + 0.5 * dt * phsp_expected.accelerations
        phsp_expected.set_positions(phsp_expected.positions + dt * v_halfstep)

        self._law.exert(phsp_expected, masses)
        phsp_expected.set_velocities(v_halfstep +
                                     0.5 * dt * phsp_expected.accelerations)

        self.assertTrue(np.allclose(phsp.positions, phsp_expected.positions),
                        msg="new position differs from expected value. "
                        f"RNG seed: {self._seed}.")

        self.assertTrue(np.allclose(phsp.velocities, phsp_expected.velocities),
                        msg="new velocity differs from expected value. "
                        f"RNG seed: {self._seed}.")

        self.assertTrue(np.allclose(phsp.accelerations,
                                    phsp_expected.accelerations),
                        msg="new acceleration differs from expected value. "
                        f"RNG seed: {self._seed}.")
