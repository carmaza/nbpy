# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines the following function:

- `write_snapshot_to_disk(options, data, time)`
  Writes the given data to a HDF5 file.

"""

import os

import h5py
import numpy.typing as npt

from nbpy.evolution import Time


def write_snapshot_to_disk(options: dict, data: npt.NDArray,
                           time: Time) -> str:
    """
    Write the given data to a HDF5 group in the given file. The data
    corresponds to variables at the given time.

    Parameters
    ----------

    `options` : dict
    Dictionary of options for observing.

    `data` : numpy.typing.NDArray
    The data, stored as a numpy array.

    `time` : nbpy.Time
    The `Time` object representing the time of observation.

    Returns
    -------

    out : string
    The _absolute_ path to the file written.

    Notes
    -----

    The argument `options` must hold the following keys:

    - `"Filename"` : string
      The name of the file to write, _without_ extension.

    - `"Groupname"`: string
      The name of the group in the HDF5 File object.

    These keys are contained, for instance, in the object returned by
    `io.input_from_yaml`, particularly the value linked to the key "Observers".

    """
    groupname = options["Groupname"]
    filename = options["Filename"]
    path = groupname + filename
    path = f"./{filename}.hdf5"

    with h5py.File(path, "a") as outfile:
        dataset_id = f"{time.id_:06}"
        outfile.create_dataset(f"{groupname}/{dataset_id}", data=data)
        outfile[groupname][dataset_id].attrs["TimeValue"] = time.value

    return os.path.abspath(path)
