# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines the function that runs the simulation:

- `run(inputfile)`

"""

import numpy as np

from nbpy import evolution
from nbpy import io


def run(inputfile: str) -> None:
    """
    Run simulation using options in YAML input file.

    Parameters
    ----------

    `inputfile` : str
    The name of the YAML input file, without any extension.

    """
    options = io.input_from_yaml(inputfile)

    particle_opts = options["Particles"]
    N = particle_opts["N"]
    masses = np.ones(N)

    evolution_opts = options["Evolution"]
    dt = evolution_opts["InitialDt"]
    integrator = evolution.Leapfrog()

    observer_opts = options["Observers"]
    observing = observer_opts["Observing"]
    group = observer_opts["Groupname"]
    filename = observer_opts["Filename"]

    # Interaction's properties.
    constant = 4. * np.pi**2.
    softening = 1.e-2
    interaction = evolution.InverseSquareLaw(constant, softening)

    # These variables will hold phase space for each timestep.
    positions = np.empty((N, 3))
    velocities = np.empty((N, 3))
    accelerations = np.empty((N, 3))

    print("Loading initial data...")
    time = evolution.Time(0, 0.)
    initial_state = evolution.RandomDistribution()
    initial_state.set_variables(positions, velocities)
    print("Initial data loaded.")

    if observing:
        filepath = io.write_snapshot_to_disk(filename, group, positions, time)
        print(f"Writing data to {filepath}")

    print("Running evolution...")
    interaction.exert(accelerations, masses, positions)
    for time_id in range(1, evolution_opts["Timesteps"]):
        integrator.evolve(positions, velocities, accelerations, dt, masses,
                          interaction)
        if observing:
            time = evolution.Time(time_id, dt * time_id)
            filepath = io.write_snapshot_to_disk(filename, group, positions,
                                                 time)
    print("Done!")
