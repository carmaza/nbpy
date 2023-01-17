# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines the function that runs the simulation:

- `run(N)`

"""

import numpy as np

from nbpy import io
from nbpy.inverse_square_law import InverseSquareLaw
from nbpy.leapfrog import Leapfrog
from nbpy.random_distribution import RandomDistribution
from nbpy.time import Time


def run(N):
    """
    Run simulation.

    Parameters
    ----------

    `N` : int
    The number of particles.

    """

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
    group = "Particles"
    filename = "Data"
    filepath = ""

    print("Loading initial data...")
    time = Time(0, 0.)
    initial_state.set_variables(positions, velocities)
    print("Initial data loaded.")
    if observing:
        filepath = io.write_snapshot_to_disk(filename, group, positions, time)
        print(f"Writing data to {filepath}")

    print("Running evolution...")
    interaction.exert(accelerations, masses, positions)
    for time_id in range(1, number_of_timesteps):
        integrator.evolve(positions, velocities, accelerations, dt, masses,
                          interaction)
        if observing:
            time = Time(time_id, dt * time_id)
            filepath = io.write_snapshot_to_disk(filename, group, positions,
                                                 time)

    print("Done!")
