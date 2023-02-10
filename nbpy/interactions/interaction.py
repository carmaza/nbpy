# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines abstract class `Interaction`.

"""
import abc

import numpy.typing as npt

from nbpy.particles import PhaseSpace


class Interaction(metaclass=abc.ABCMeta):
    """
    Abstract base class for interactions.

    """

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, params: dict):
        """
        Constructor that uses a dictionary of parameters.

        """

    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        """
        The name of the interaction.

        """

    @abc.abstractmethod
    def exert(self, phsp: PhaseSpace, masses: npt.NDArray) -> None:
        """
        Set the accelerations in terms of the phase space.

        """
