# Distributed under the MIT License.
# See LICENSE for details.

from context import nbpy

import numpy as np
import unittest

from nbpy.inverse_square_law import InverseSquareLaw


class TestInverseSquareLaw(unittest.TestCase):
    """
    Test functions in `InverseSquareLaw` class.
    """

    @staticmethod
    def name():
        return "TestInverseSquareLaw"

    def test(self):

        seed = np.random.randint(0, 1e6)
        np.random.seed(seed)

        dim = 3
        n_body = np.random.randint(2, 10)
        positions = np.random.randn(n_body, dim)
        masses = np.random.rand(n_body)

        constant = np.random.rand()
        softening = np.random.rand()
        law = InverseSquareLaw(constant, softening)
        acceleration = law.acceleration(positions, masses)

        acceleration_expected = np.zeros_like(positions)
        for j in range(n_body):
            for k, mass in enumerate(masses):
                d_x = positions[k, 0] - positions[j, 0]
                d_y = positions[k, 1] - positions[j, 1]
                d_z = positions[k, 2] - positions[j, 2]
                d_cube = (d_x**2. + d_y**2. + d_z**2. + softening**2.)**1.5
                acceleration_expected[j, 0] += mass * d_x / d_cube
                acceleration_expected[j, 1] += mass * d_y / d_cube
                acceleration_expected[j, 2] += mass * d_z / d_cube

        self.assertTrue(
            np.allclose(acceleration, acceleration_expected),
            msg="In {name}: acceleration differs from expected value. "
            "RNG seed: {seed}.".format(name=self.name(), seed=seed))

        print("\nAll tests in {s} passed.".format(s=self.name()))


if __name__ == "__main__":
    unittest.main()
