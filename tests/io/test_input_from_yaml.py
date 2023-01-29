# Distributed under the MIT License.
# See LICENSE for details.
"""
Contains unit tests for `io.input_from_yaml`.

"""

import unittest

from nbpy import io


class TestInputFromYaml(unittest.TestCase):
    """
    Test function `io.input_from_yaml`.

    """

    def test_implementation(self):
        """
        Test general writing using a temporary file.

        """
        filepath = "./tests/io"
        options = io.input_from_yaml(f"{filepath}/Example")

        expected = {}
        expected["Particles"] = {"N": 50}
        expected["Evolution"] = {"InitialDt": 1.e-3, "Timesteps": 10}
        expected["Observers"] = {
            "Observing": True,
            "Filename": "Data",
            "Groupname": "Particles"
        }

        self.assertEqual(
            options,
            expected,
            msg="options read from YAML file differ from expected value. ")
