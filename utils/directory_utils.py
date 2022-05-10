from errno import EEXIST
from os import makedirs,path

def mkdir_p(mypath):
    """
    Creates a directory and, if needed, all parent directories.
    """
    try:
        makedirs(mypath)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise