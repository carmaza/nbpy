# Distributed under the MIT License.
# See LICENSE for details.

import numpy as np

from inverse_square_law import InverseSquareLaw
from leapfrog import Leapfrog
from random_distribution import RandomDistribution


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
    observe = True

    print("Loading initial data...")
    initial_state.set_variables(positions, velocities)
    interaction.exert(accelerations, masses, positions)
    print("Initial data loaded.")

    print("Running evolution...")
    for i in range(1, number_of_timesteps):
        integrator.evolve(positions, velocities, accelerations, dt, masses,
                          interaction)

    print("Done!")
