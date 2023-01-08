# Distributed under the MIT License.
# See LICENSE for details.

import os

import matplotlib.pyplot as plt
import numpy as np

import nbpy.plot as plot

from nbpy.inverse_square_law import InverseSquareLaw
from nbpy.leapfrog import Leapfrog
from nbpy.random_distribution import RandomDistribution


def run(N):

    # Particles' properties.
    masses = np.ones(N)
    initial_state = RandomDistribution()

    # Interaction's properties.
    constant = 4. * np.pi**2.
    softening = 1.e-4
    interaction = InverseSquareLaw(constant, softening)

    # Initial values.
    positions = np.empty((N, 3))
    velocities = np.empty((N, 3))
    accelerations = np.empty((N, 3))

    # Evolution parameters.
    dt = 1.e-3
    number_of_timesteps = 10
    integrator = Leapfrog()

    # Observe parameters.
    observing = True
    figvol = plt.figure()
    axvol = plt.axes(projection='3d')
    if observing:
        figs_folder = "figs"
        try:
            os.mkdir("./{}".format(figs_folder))
            print("Local folder /{} created.".format(figs_folder))
        except:
            print("Local folder /{} already present.".format(figs_folder))

    print("Loading initial data...")
    initial_state.set_variables(positions, velocities)
    if observing:
        plot.positions_3d(figvol, axvol, 0, dt, positions)
    print("Initial data loaded.")

    print("Running evolution...")
    interaction.exert(accelerations, masses, positions)
    for time_id in range(1, number_of_timesteps):
        integrator.evolve(positions, velocities, accelerations, dt, masses,
                          interaction)
        if observing:
            plot.positions_3d(figvol, axvol, time_id, dt, positions)

    plt.close(figvol)
    print("Done!")
