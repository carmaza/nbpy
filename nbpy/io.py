# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines the following Input/Output functions:

- `write_snapshot_to_disk(filename, groupname, data, time)`
  Writes the given data to a HDF5 file.

"""

import os

import h5py
import numpy.typing as npt

from nbpy.time import Time


def write_snapshot_to_disk(filename: str, groupname: str, data: npt.NDArray,
                           time: Time) -> str:
    """
    Write the given data to a HDF5 group in the given file. The data
    corresponds to variables at the given time.

    Parameters
    ----------

    `filename` : string
    The name of the file to write, _without_ extension.

    `groupname`: string
    The name of the group in the HDF5 File object.

    `data` : numpy.typing.NDArray
    The data, stored as a numpy array.

    `time` : nbpy.Time
    The `Time` object representing the time of observation.

    Returns
    -------

    out : string
    The _absolute_ path to the file written.

    """
    path = groupname + filename
    path = f"./{filename}.hdf5"
    with h5py.File(path, "a") as outfile:
        dataset_id = f"{time.id_:06}"
        outfile.create_dataset(f"{groupname}/{dataset_id}", data=data)
        outfile[groupname][dataset_id].attrs["TimeValue"] = time.value
    return os.path.abspath(path)
