# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines the function :func:`.write_snapshot_to_disk`.

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

    options : dict
        Dictionary of options for observing.

    data : numpy.typing.NDArray
        The data, stored as a numpy array.

    time : :class:`.Time`
        The object representing the time of observation.

    Returns
    -------

    out : string
        The absolute path to the file written.

    Notes
    -----

    The argument ``options`` must hold the following keys:

    - ``"Filename"``: the name of the file to write, without extension.
    - ``"Groupname"``: the name of the group in the HDF5 File object.

    These keys are contained, for instance, in the object returned by
    :func:`.input_from_yaml`, particularly the value linked to the key
    ``"Observers"``.

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
