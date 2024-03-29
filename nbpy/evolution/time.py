# Distributed under the MIT License.
# See LICENSE for detais.
"""
Defines class :class:`.Time`.

"""


class Time:
    """
    A time instant.

    Parameters
    ----------

    id_ : int
        A unique ID.

    value : float
        The numerical value of the time instant.

    """

    def __init__(self, id_: int, value: float):
        """
        Parameters
        ----------

        id_ : int
            The unique ID.

        value : float
            The time value.

        """
        self._id = id_
        self._value = value

    @property
    def id_(self) -> int:
        """
        The time ID.

        """
        return self._id

    @property
    def value(self) -> float:
        """
        The time value.

        """
        return self._value
