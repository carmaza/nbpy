# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines class `Leapfrog`.

"""

import numpy.typing as npt

from nbpy.particles import PhaseSpace


class Leapfrog:
    """
    The syncronized second-order Leapfrog integrator for oscillatory problems.

    Functions
    ---------

    `evolve(phsp, dt, masses, interaction)`
    The algorithm to update the phase space.

    """

    @staticmethod
    def evolve(phsp: PhaseSpace, dt: float, masses: npt.NDArray,
               interaction) -> None:
        """
        Update positions and velocities.

        Since it is needed in the calculation, this function also updates the
        accelerations for the given interaction.

        Parameters
        ----------

        `phsp` : nbpy.particles.Particles [mutates]
        The phase space of the system. Must contain keys "Positions",
        "Velocities", and "Accelerations".

        `dt` : float
        The (fixed) time step. Must be lower than twice the characteristic
        oscillation period.

        `masses` : numpy.typing.NDArray
        The masses of the particles.

        `interaction` : obj
        The interaction from which to calculate the acceleration in terms of the
        position. Must have an `exert(phsp, masses)` member function.

        """
        phsp.set_velocities(phsp.velocities + 0.5 * dt * phsp.accelerations)
        phsp.set_positions(phsp.positions + dt * phsp.velocities)
        interaction.exert(phsp, masses)
        phsp.set_velocities(phsp.velocities + 0.5 * dt * phsp.accelerations)
