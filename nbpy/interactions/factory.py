# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines class :class:`.Interactions`.

"""

from typing import Type

from .interaction import Interaction
from .inverse_square_law import InverseSquareLaw


class Interactions:
    """
    A class to obtain all available types derived from
    :class:`.Interaction` s.

    """

    @staticmethod
    def typelist() -> list[Type[Interaction]]:
        """
        Return a list of all available subtypes.

        """
        return [InverseSquareLaw]

    @classmethod
    def typedict(cls) -> dict[str, Type[Interaction]]:
        """
        Return a dictionary of all available subtypes, where the keys
        are the subtype names, and the values the types.

        """
        return {t.name(): t for t in cls.typelist()}
