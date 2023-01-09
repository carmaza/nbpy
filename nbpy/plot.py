# Distributed under the MIT License.
# See LICENSE for details.

import matplotlib.pyplot as plt


def positions_3d(axis, time, positions, folder, grid_off=True):
    axis.scatter(positions[:, 0],
                 positions[:, 1],
                 positions[:, 2],
                 color='white',
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
