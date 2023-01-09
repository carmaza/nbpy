# Distributed under the MIT License.
# See LICENSE for detais.
"""
Defines the class Time.
"""


class Time:
    """
    A time instant.

    Attributes
    ----------
    `time_id` : int
    A unique ID.

    `value` : float
    The numerical value of the time instant.
    """

    def __init__(self, id: int, value: float):
        self._id = id
        self._value = value

    @property
    def id(self) -> int:
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
