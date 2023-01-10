# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines the following functions to plot simulation data:
- positions_3d(axis, time, positions, folder, grid_off=True)

"""
import matplotlib.pyplot as plt


def positions_3d(axis,
                 time,
                 positions,
                 folder,
                 center_of_mass=None,
                 grid_off=True):
    """
    Plot particles at their given 3d positions at the given time.

    Parameters
    ----------
    `axis` : obj
    The Matplotlib object containing the plot axis.

    `time` : obj
    The Time object representing the time of observation.

    `positions` : ndarray
    The positions at the given time, stored as a N-by-3 NumPy array.

    `folder` : string
    The local folder where to store the plot.

    `center_of_mass` : ndarray (default: None)
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
