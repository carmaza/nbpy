# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines the following function:

- `input_from_yaml(inputfile)`
  Reads input file and returns dict containing simulation options.

"""

import errno

import yaml


def input_from_yaml(inputfile: str) -> dict:
    """
    Return dictionary of simulations options specified in YAML input file.

    The options specified in the input file are:

    ```
    Particles:
      N: <int>

    Evolution:
      InitialDt: <float>
      Timesteps: <int>

    Observers:
      Observing: <bool>
      Filename: <str>
      Groupname: <str>

    ```

    Parameters
    ----------

    `inputfile` : str
    The name of the target input file, without the YAML extension.

    Returns
    -------

    out : `dict`
    A dictionary of the options for the simulation.

    """
    try:
        with open(f"{inputfile}.yml", "r", encoding="utf-8") as infile:
            return yaml.safe_load(infile)
    except OSError as err:
        if err.errno == errno.ENOENT:
            print(f"Local file '{inputfile}.yml' does not exist!")
        else:
            print(f"Unknown error when loading {inputfile}.")
        raise
