# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 00:38:46 2021

@author: Christian Konstantinov
"""

from abc import ABC
from dataclasses import dataclass, asdict, replace
import json

@dataclass
class HyperParameters(ABC):

    """Abstract Base Class for storing and accessing hyperparameters."""

    def __post_init__(self):
        """Make sure that the abstract class is never instantiated."""
        if self.__class__ == HyperParameters:
            raise TypeError("Cannot instantiate abstract class.")

@dataclass
class DSP(HyperParameters):

    """A dataclass for storing and accessing signal processing hyperparameters."""

    fs: int = 8000 # sample rate
    W: int = 1024 # fft window size
    stride: int = int(0.001 * fs)
    bands: int = 20 # number of frequency bins for the spectrogram
    f_min: float = 20.0 # Humans cannot hear below 20 Hz
    f_max: float = 0.5 * fs # Nyquist frequency
    context: int = int(0.15 * fs / stride) # tensor width in fft frames
    tolerance: int = int(0.02 * fs / stride) # margin of error in fft frames

@dataclass
class ML(HyperParameters):

    """Dataclass for storing and accessing machine learning hyperparameters."""

    input_size: int = DSP.bands # a bit wonky, but it works
    sequence_length: int = DSP.context
    hidden_size: int = 32
    num_layers: int = 2
    num_classes: int = 2
    learning_rate: float = 0.0001
    batch_size: int = 128
    num_epochs: int = 100
    num_workers: int = 6
    patience: int = 20

def save(h: HyperParameters, file_path: str) -> None:
    """Given a file path, save the current Hyperparameters to a json file."""
    with open(file_path, 'w+') as f:
        json.dump(asdict(h), f)

def load(h: HyperParameters, file_path: str) -> HyperParameters:
    """Given a file path, load the Hyperparameters from file into the class."""
    with open(file_path, 'r') as f:
        h = replace(h, **json.load(f))
    return h
