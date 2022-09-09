# Distributed under the MIT License.
# See LICENSE for details.


class Leapfrog:
    """
    The syncronized second-order Leapfrog integrator for oscillatory problems.
    """

    @staticmethod
    def evolve(dt, pos, vel, acc, interaction, *argv):
        """
        The algorithm to update the evolved variables.

        Parameters
        ----------

        `dt` : float
        The (fixed) time step. Must be lower than twice the oscillation period.

        `pos, vel, acc` : ndarray, ndarray, ndarray
        The positions, velocities, and accelerations of the system. They must
        have the same array shape.

        `interaction` : obj
        The interaction from which to calculate the acceleration in terms of the
        position. Must have an `acceleration(pos, *argv)` member function.

        `*argv`
        Any additional arguments of `interaction.acceleration`.

        Returns
        -------

        out : ndarray, ndarray, ndarray
        The updated positions, velocities and accelerations.

        """
        new_vel = vel + 0.5 * dt * acc
        new_pos = pos + dt * new_vel
        new_acc = interaction.acceleration(new_pos, *argv)
        new_vel += 0.5 * dt * new_acc
        return new_pos, new_vel, new_acc
