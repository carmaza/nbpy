# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines the following utility functions:

- `create_folder(folder)`
  Creates a folder and throws exception if it can't.

"""

import errno
import os


def create_folder(path: str) -> None:
    """
    Create folder given the path.

    Parameters
    ----------

    `path` : str
    The path to the new folder, including its name.

    """
    try:
        os.makedirs(f"{path}")
        print(f"Folder {path} created.")
    except OSError as err:
        if err.errno == errno.EEXIST:
            print(f"Local folder {path} already exists.")
        else:
            raise
