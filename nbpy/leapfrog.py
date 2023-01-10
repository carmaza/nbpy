# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines class `Leapfrog`.

"""


class Leapfrog:
    """
    The syncronized second-order Leapfrog integrator for oscillatory problems.

    Functions
    ---------

    `evolve(pos, vel, acc, dt, masses, interaction)`
    The algorithm to update the positions and velocities.

    """

    @staticmethod
    def evolve(pos, vel, acc, dt, masses, interaction):
        """
        Update positions and velocities.

        Since it is needed in the calculation, this function also updates the
        accelerations for the given interaction.

        Parameters
        ----------

        `pos, vel, acc` : ndarray, ndarray, ndarray [mutate]
        The positions, velocities, and accelerations of the system. They must
        have the same array shape among each other.

        `dt` : float
        The (fixed) time step. Must be lower than twice the characteristic
        oscillation period.

        `masses` : ndarray
        The masses of the particles.

        `interaction` : obj
        The interaction from which to calculate the acceleration in terms of the
        position. Must have an `exert(acc, masses, pos)` member function.

        """
        vel += 0.5 * dt * acc
        pos += dt * vel
        interaction.exert(acc, masses, pos)
        vel += 0.5 * dt * acc
