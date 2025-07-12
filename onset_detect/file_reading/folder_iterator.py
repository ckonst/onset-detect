import os
from glob import glob
from typing import Generator


def subdir_names(root: str) -> Generator[str, None, None]:
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
