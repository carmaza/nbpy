# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines the function that runs the simulation:

- `run(inputfile)`

"""

import numpy as np

from nbpy import evolution, interactions, io, particles


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
    timesteps = evolution_opts["Timesteps"]
    integrator = evolution.Leapfrog()

    observer_opts = options["Observers"]
    observing = observer_opts["Observing"]

    # Interaction's properties.
    constant = 4. * np.pi**2.
    softening = 1.e-2
    interaction = interactions.InverseSquareLaw(constant, softening)

    # Holds phase space variables: positions, velocities, accelerations.
    phsp = particles.PhaseSpace(N)

    print("Loading initial data...")
    time = evolution.Time(0, 0.)
    initial_state = evolution.RandomDistribution()
    initial_state.set_variables(phsp)
    print("Initial data loaded.")

    # With the initial conditions set, calculate initial accelerations.
    interaction.exert(phsp, masses)

    if observing:
        filepath = io.write_snapshot_to_disk(observer_opts, phsp.positions,
                                             time)
        print(f"Writing data to {filepath}")

    print("Running evolution...")
    for time_id in range(1, timesteps):
        integrator.evolve(phsp, dt, masses, interaction)
        if observing:
            time = evolution.Time(time_id, dt * time_id)
            io.write_snapshot_to_disk(observer_opts, phsp.positions, time)

    print("Done!")
