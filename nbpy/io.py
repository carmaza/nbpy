# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines the following Input/Output functions:

- `write_snapshot_to_disk(filename, groupname, data, time)`
  Write the given data to a HDF5 file.

"""

import os

import h5py


def write_snapshot_to_disk(filename, groupname, data, time):
    """
    Writes the given data to a HDF5 file. The data corresponds to variables at
    the given time.

    Parameters
    ----------

    `filename` : string
    The name of the file to write, _without_ extension.

    `groupname`: string
    The name of the group in the HDF5 File object.

    `data` : ndarray
    The data, stored as a numpy array.

    `time` : obj
    The `Time` object representing the time of observation.

    Returns
    -------

    out : string
    The _absolute_ path to the file written.

    """
    path = f"./{filename}.hdf5"
    with h5py.File(path, "a") as outfile:
        outfile.create_dataset(f"{groupname}/{time.id_:06}", data=data)
    return os.path.abspath(path)
