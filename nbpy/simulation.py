# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines the function that runs the simulation:

- `run(N)`

"""

import numpy as np

from nbpy import evolution
from nbpy import io


def run(N: int) -> None:
    """
    Run simulation.

    Parameters
    ----------

    `N` : int
    The number of particles.

    """

    # Particles' properties.
    masses = np.ones(N)
    initial_state = evolution.RandomDistribution()

    # Interaction's properties.
    constant = 4. * np.pi**2.
    softening = 1.e-2
    interaction = evolution.InverseSquareLaw(constant, softening)

    # Initial values.
    positions = np.empty((N, 3))
    velocities = np.empty((N, 3))
    accelerations = np.empty((N, 3))

    # Evolution parameters.
    dt = 1.e-3
    number_of_timesteps = 10
    integrator = evolution.Leapfrog()

    # Observe parameters.
    observing = True
    group = "Particles"
    filename = "Data"
    filepath = ""

    print("Loading initial data...")
    time = evolution.Time(0, 0.)
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
            time = evolution.Time(time_id, dt * time_id)
            filepath = io.write_snapshot_to_disk(filename, group, positions,
                                                 time)

    print("Done!")
