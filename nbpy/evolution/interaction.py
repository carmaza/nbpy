# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines abstract class `Interaction`.

"""
import abc

import numpy.typing as npt


class Interaction(metaclass=abc.ABCMeta):
    """
    Abstract base class for interactions.

    """

    @abc.abstractmethod
    def exert(self, accelerations: npt.NDArray, masses: npt.NDArray,
              positions: npt.NDArray) -> None:
        """
        Calculate the accelerations in terms of the phase space.
        """
