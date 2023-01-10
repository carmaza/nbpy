# Distributed under the MIT License.
# See LICENSE for details.

import matplotlib.pyplot as plt
import numpy as np

import nbpy.plot as plot
import nbpy.util as util
import nbpy.phase_space as phase_space

from nbpy.inverse_square_law import InverseSquareLaw
from nbpy.leapfrog import Leapfrog
from nbpy.random_distribution import RandomDistribution
from nbpy.time import Time


def run(N, figure_folder="figures"):

    # Particles' properties.
    masses = np.ones(N)
    initial_state = RandomDistribution()

    # Interaction's properties.
    constant = 4. * np.pi**2.
    softening = 1.e-2
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
        util.create_folder(figure_folder)

    print("Loading initial data...")
    time = Time(0, 0.)
    initial_state.set_variables(positions, velocities)
    if observing:
        center_of_mass = phase_space.center_of_mass(masses, positions)
        plot.positions_3d(axvol, time, positions, figure_folder,
                          center_of_mass)
    print("Initial data loaded.")

    print("Running evolution...")
    interaction.exert(accelerations, masses, positions)
    for time_id in range(1, number_of_timesteps):
        integrator.evolve(positions, velocities, accelerations, dt, masses,
                          interaction)
        if observing:
            time = Time(time_id, dt * time_id)
            center_of_mass = phase_space.center_of_mass(masses, positions)
            plot.positions_3d(axvol, time, positions, figure_folder,
                              center_of_mass)

    plt.close(figvol)
    print("Done!")
