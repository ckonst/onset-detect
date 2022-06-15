from glob import glob
import os

def iterate_folder(root: str) -> str:
    """Given a path to a directory, yield the name of each subdirectory.

    Parameters
    ----------
    root : str
        The path to the top-level folder.

    Yields
    ------
    str
        The name of the next directory in root.

    """
    for folder in glob(f'{root}/*/'):
        yield folder.split(os.path.sep)[-2]