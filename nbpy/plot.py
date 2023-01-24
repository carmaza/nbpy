# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines the following functions to plot simulation data:

- `orbits_3d(filepath, groupname, figure_folder="figures")`
  Plots time evolution of the orbits of the particles.

- `positions_3d(axis, time, positions, folder, grid_off=True)`
  Plots particles at their 3-d positions at a specific time.

"""

from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy.typing as npt

from nbpy import util
from nbpy.time import Time


def positions_3d(axis,
                 time: Time,
                 positions: npt.NDArray,
                 folder: str,
                 center_of_mass: Optional[npt.NDArray] = None,
                 grid_off: bool = True) -> None:
    """
    Plot particles at their 3-d positions at the given time.

    Parameters
    ----------

    `axis` : obj
    The Matplotlib object containing the plot axis.

    `time` : nbpy.Time
    The Time object representing the time of observation.

    `positions` : numpy.typing.NDArray
    The positions at the given time, stored as a N-by-3 NumPy array.

    `folder` : string
    The local folder where to store the plot.

    `center_of_mass` : numpy.typing.NDArray (default: None)
    If given, then plot it along with the system.

    `grid_off` : bool (default: True)
    Whether to turn off plot grid.

    """

    axis.scatter(positions[:, 0],
                 positions[:, 1],
                 positions[:, 2],
                 color='white',
                 depthshade=False,
                 s=1.)

    if center_of_mass is not None:
        axis.scatter(center_of_mass[0],
                     center_of_mass[1],
                     center_of_mass[2],
                     color='red',
                     depthshade=False,
                     s=1.)

    half_side = 2.
    axis.set_xlim(-half_side, half_side)
    axis.set_ylim(-half_side, half_side)
    axis.set_zlim(-half_side, half_side)

    if grid_off:
        axis.grid(False)

    axis.set_xticks([-0.5, 0.5])
    axis.set_xticklabels(["astronomical unit", ""])
    axis.set_yticks([])
    axis.set_zticks([])

    axis.yaxis.labelpad = 12
    axis.yaxis.set_rotate_label(False)

    black = (0., 0., 0.)
    background_color = black
    axis.xaxis.set_pane_color(background_color + (0.95, ))
    axis.yaxis.set_pane_color(background_color + (0.9, ))
    axis.zaxis.set_pane_color(background_color)

    plt.title(f"elapsed time: {time.value:1.3f} years")

    plt.savefig(f"{folder}/{time.id_:06}", bbox_inches='tight', dpi=300)

    axis.clear()


def orbits_3d(filepath: str,
              groupname: str,
              figure_folder: str = "./figures") -> None:
    """
    Plot time evolution of the orbits as a sequence of snapshots.

    Parameters
    ----------

    `filepath` : str
    The path to the file containing the data.

    `groupname` : str
    The HDF5 group name in the file.

    `figure_folder` : str (default: "./figures")
    The local folder where to store the figures.

    """
    fig = plt.figure()
    axis = plt.axes(projection='3d')
    util.create_folder(figure_folder)

    with h5py.File(filepath, "r") as readfile:
        for key, value in readfile[groupname].items():
            time = Time(key, value.attrs["TimeValue"])
            positions_3d(axis, time, value, figure_folder)

    plt.close(fig)
