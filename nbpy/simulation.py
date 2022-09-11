# Distributed under the MIT License.
# See LICENSE for details.

import numpy as np

from inverse_square_law import InverseSquareLaw
from leapfrog import Leapfrog
from random_distribution import RandomDistribution


def run():
    # Particles' properties.
    N = 2
    masses = np.ones(N)
    initial_state = RandomDistribution()

    # Interaction's properties.
    constant = 1.
    softening = 1.e-4
    interaction = InverseSquareLaw(constant, softening)

    # Initial values.
    positions = np.empty((N, 3))
    velocities = np.empty((N, 3))
    accelerations = np.empty((N, 3))

    print("Loading initial data...")
    initial_state.set_variables(positions, velocities)
    interaction.exert(accelerations, masses, positions)
    print("Initial data loaded.")

    # Evolution parameters.
    dt = 1.e-4
    Nt = 600000
    integrator = Leapfrog()

    print("Running evolution...")
    for i in range(1, Nt):
        integrator.evolve(positions, velocities, accelerations, dt, masses,
                          interaction)

    print("Done!")
