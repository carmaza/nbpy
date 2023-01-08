# Distributed under the MIT License.
# See LICENSE for details.

import matplotlib.pyplot as plt


def positions_3d(fig,
                 ax,
                 time_id,
                 dt,
                 positions,
                 folder="",
                 zfill=6,
                 axis_off=False,
                 grid_off=True,
                 ticks_off=True):

    ax.scatter(positions[:, 0],
               positions[:, 1],
               positions[:, 2],
               color='white',
               depthshade=False,
               s=1.)

    half_side = 2.
    ax.set_xlim(-half_side, half_side)
    ax.set_ylim(-half_side, half_side)
    ax.set_zlim(-half_side, half_side)

    if grid_off:
        ax.grid(False)

    if ticks_off:
        ax.set_xticks([-0.5, 0.5])
        ax.set_xticklabels(["astronomical unit", ""])
        ax.set_yticks([])
        ax.set_zticks([])

    ax.yaxis.labelpad = 12
    ax.yaxis.set_rotate_label(False)
    if axis_off:
        ax.set_axis_off()

    black = (0., 0., 0.)
    background_color = black
    ax.xaxis.set_pane_color(background_color + (0.95, ))
    ax.yaxis.set_pane_color(background_color + (0.9, ))
    ax.zaxis.set_pane_color(background_color)

    plt.title("elapsed time: {:1.3f} years".format(time_id * dt))

    plt.savefig("figs/{}/{}".format(folder,
                                    str(time_id).zfill(zfill)),
                bbox_inches='tight',
                dpi=300)

    ax.clear()
