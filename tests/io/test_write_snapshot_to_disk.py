# Distributed under the MIT License.
# See LICENSE for details.
"""
Contains unit tests for `io.write_snapshot_to_disk`.

"""

import os
import unittest

import h5py
import numpy as np

from nbpy import io
from nbpy.evolution import Time


class TestWriteSnapshotToDisk(unittest.TestCase):
    """
    Test `io.write_snapshot_to_disk`.

    """

    def test(self):
        """
        Test general writing using a temporary file.

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
            dataset_written = readfile[groupname][f"{id_:06}"]
            self.assertTrue(np.allclose(data_to_write, dataset_written[:]),
                            msg="dataset written differs from dataset read. "
                            f"RNG seed: {seed}.")
            self.assertAlmostEqual(
                dataset_written.attrs["TimeValue"],
                time.value,
                msg="time value differs from expected value. "
                f"RNG seed: {seed}.")

        os.remove(filepath)
