# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 00:38:46 2021

@author: Christian Konstantinov
"""

from dataclasses import dataclass
import pickle

@dataclass
class Hyperparameters:
    """Class for storing hyperparameters and writing them to disk."""

    input_size: int = 1024
    hidden_size: int = 256
    num_layers: int = 3
    num_classes: int = 2
    learning_rate: float = 0.001
    batch_size: int = 128
    num_epochs: int = 10

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

    def save(self, file_path):
        with open(file_path, 'wb+') as f:
            pickle.dump(self.__dict__, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.__dict__.update(pickle.load(f))