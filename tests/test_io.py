# Distributed under the MIT License.
# See LICENSE for details.
"""
Contains unit tests for functions in modio `io`.

"""

import os
import unittest

import h5py
import numpy as np

from nbpy import io
from nbpy.time import Time


class TestIO(unittest.TestCase):
    """
    Test functions in module `io`.

    """

    def test_write_to_disk(self):
        """
        Test function `write_snapshot_to_disk`.

        """

        seed = np.random.randint(0, 1e6)
        np.random.seed(seed)

        N = np.random.randint(1, 10)
        data_to_write = np.random.randn(N, 3)

        id_ = np.random.randint(1, 10)
        time = Time(id_, np.random.randn())
        groupname = "Temp"
        filepath = io.write_snapshot_to_disk("temp", groupname, data_to_write,
                                             time)

        with h5py.File(filepath, "r") as readfile:
            data_written = readfile[groupname][f"{id_:06}"][:]
            self.assertTrue(np.allclose(data_to_write, data_written),
                            msg="time value differs from expected value. "
                            f"RNG seed: {seed}.")

        os.remove(filepath)
