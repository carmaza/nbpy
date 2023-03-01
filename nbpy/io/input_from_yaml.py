# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines the function :func:`.input_from_yaml`.

"""

import errno

import yaml


def input_from_yaml(inputfile: str) -> dict:
    """
    Return dictionary of simulations options specified in YAML input file.

    The options specified in the input file follow the template

    .. code-block::

        Particles:
          N: (int)

        Interaction:
          <name>:
            <parameter>: ...
            <parameter>: ...
            ...

        Evolution:
          InitialDt: (float)
          Timesteps: (int)

        Observers:
          Observing: (bool)
          Filename: (str)
          Groupname: (str)

    An example can be found  ``/tests/io/Example.yml``. See
    ``/interactions`` for all available interactions. 

    Parameters
    ----------

    inputfile : str
        The name of the target input file, without the YAML extension.

    Returns
    -------

    out : dict
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
