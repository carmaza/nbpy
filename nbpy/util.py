# Distributed under the MIT License.
# See LICENSE for details.

import errno
import os


def create_folder(folder):
    try:
        os.makedirs("./{}".format(folder))
        print("Local folder /{} created.".format(folder))
    except OSError as e:
        if (e.errno == errno.EEXIST):
            print("Local folder /{} already exists.".format(folder))
        else:
            raise
