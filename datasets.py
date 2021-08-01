# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 22:29:13 2021

@author: Christian Konstantinov
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple

from folder_iterator import iterate_folder

from functools import wraps
from time import time

def measure(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print('Elapsed time: {} ms'.format((end-start) * 1000))
        return result
    return wrapper

DATA_PATH = './dataset/osu'
EXTRACT_PATH = f'{DATA_PATH}/extracted'

def get_size() -> int:
    """Get the total size of the dataset."""
    return sum([len(torch.load(f'{EXTRACT_PATH}/{name}/features.pt')[1])
        for name in iterate_folder(EXTRACT_PATH)])

def get_index_table() -> Dict[int, Tuple[str, int]]:
    """Create a mapping from Dataset index (int) to name (str) and context frame index (int)."""
    index_table = {}
    dataset_index = 0
    for name in iterate_folder(EXTRACT_PATH):
        _, indices = torch.load(f'{EXTRACT_PATH}/{name}/features.pt')
        for i in indices:
            index_table[dataset_index + i] = (name, i)
        dataset_index += len(indices)
    return index_table

class OnsetDataset(Dataset):
    """Dataset class for mapping spectrogram frames to onset classes."""

    def __init__(self, **kwargs):
        """Useful docstring goes here."""
        self.__dict__.update(**kwargs)
        self._size = get_size()
        self.index_table = get_index_table()

    def __len__(self) -> int:
        """Useful docstring goes here."""
        return self._size

    def __getitem__(self, index: int) -> torch.Tensor:
        """Useful docstring goes here."""
        name, frame = self.index_table[index]
        tensor, indices = torch.load(f'{EXTRACT_PATH}/{name}/features.pt')
        targets = torch.load(f'{EXTRACT_PATH}/{name}/targets.pt')
        context = tensor.shape[-1] // len(indices)
        start = frame * context
        end = (frame + 1) * context
        frame = torch.FloatTensor([frame])
        return ((tensor[:, :, start:end], frame), targets[start:end])

class CoordinateDataset(Dataset):
    """Dataset class for mapping Onset times to x, y coordinates."""
    # TODO: Finish implementing this class

    def __init__(self, **kwargs):
        """Useful docstring goes here."""
        self.__dict__.update(kwargs)

    def __len__(self):
        """Useful docstring goes here."""
        pass

    def __getitem__(self, index):
        """Useful docstring goes here."""
        pass
