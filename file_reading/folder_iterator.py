from glob import glob
import os

def iterate_folder(root):
    for folder in glob(f'{root}/*/'):
        yield folder.split(os.path.sep)[-2]