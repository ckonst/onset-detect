# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 14:35:39 2021.

@author: Christian Konstantinov
"""

import json
import librosa as lb

from glob import glob
from typing import Tuple

import torch
from torchaudio import transforms
from torch.multiprocessing import Pool

from evaluation.time_elapsed import timed
from file_reading.folder_iterator import iterate_folder
from model.hyperparameters import DSP, ML


DATA_PATH = './dataset/osu'
RAW_PATH = f'{DATA_PATH}/raw'
EXTRACT_PATH = f'{DATA_PATH}/extracted'

# TODO:
# - Add log mel / STFT autocorrelation = |G(x)|² preprocessing
# - Multiple spectrogram feature processing
# - Maybe tempo estimation

@timed(units='s')
def extract() -> None:
    """Create processes to parallelize feature extraction."""
    inputs = [name for name in iterate_folder(RAW_PATH)]
    ml = ML()
    with Pool(ml.num_workers) as p:
        p.map(process_and_save_song, inputs)

def process_and_save_song(name: str) -> None:
    """Extract features from a given song name, then save them to the new folder."""
    (tensor, indices), targets = process_song(name)
    song_path = f'{EXTRACT_PATH}/{name}'
    torch.save((tensor, indices), f'{song_path}/features.pt')
    torch.save(targets, f'{song_path}/targets.pt')

def process_song(name: str) -> Tuple[Tuple[torch.Tensor, int], torch.Tensor]:
    """Process one song and return target and feature Tensors with the index to the spectrogram frame."""
    dsp = DSP()
    song_path = f'{EXTRACT_PATH}/{name}'
    tensor, indices = extract_features(name, dsp)
    targets = create_onset_labels(tensor, song_path, dsp)
    return (tensor, indices), targets

def extract_features(name: str, dsp: DSP) -> torch.Tensor:
    """Extract the spectrogams and tensor indices for the dataset."""
    tensor = get_lmfs(name, dsp)
    tensor = pad_tensor(tensor, tensor.shape[-1], dsp.context)
    indices = list(range(0, tensor.shape[-1] // dsp.context))
    return tensor, indices

def create_onset_labels(features: torch.Tensor, song_path: str, dsp: DSP) -> torch.Tensor:
    """Given the features (spectrogram) of the data, return the target onsets."""
    with open(f'{song_path}/beatmap.json', 'r') as f:
        beatmap = json.load(f)
    onsets = lb.time_to_frames(beatmap['onsets'], sr=dsp.fs, hop_length=dsp.stride)
    targets = torch.zeros(features.shape[2])
    for o in onsets:
        targets[o] = 1
    return targets

def get_lmfs(name: str, dsp: DSP) -> torch.Tensor:
    """Return the log mel frequency spectrogram."""
    map_path = f'{RAW_PATH}/{name}'
    mono_sig, fs = lb.load(glob(f'{map_path}/*.mp3')[0], sr=dsp.fs, res_type='kaiser_fast')
    mono_sig = torch.from_numpy(mono_sig)
    norm_sig = normalize(mono_sig)

    mfs = transforms.MelSpectrogram(sample_rate=fs, n_fft=dsp.W,
                                    f_min=dsp.f_min, f_max=dsp.f_max,
                                    n_mels=dsp.bands, hop_length=dsp.stride,
                                    window_fn=torch.hamming_window)(norm_sig)

    lmfs = transforms.AmplitudeToDB()(mfs).unsqueeze(0).half().detach()
    return lmfs

def pad_tensor(unpadded: torch.Tensor, size: int, W: int) -> torch.Tensor:
    """Return the input tensor padded to be the length of the nearest mulitple of W."""
    pad_sig = torch.zeros(unpadded.shape[0], unpadded.shape[1], size + (W - (size % W)))
    pad_sig[:, :, :size] = unpadded
    return pad_sig

def normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Return the tensor normalized to the interval [-1, 1] where μ = 0, σ² = 1."""
    minus_mean = tensor - tensor.float().mean()
    return minus_mean / minus_mean.abs().max()

def create_coordinate_labels():
    """Return the coordinate label dataset."""
    targets = []
    for name in iterate_folder(EXTRACT_PATH):
        path = f'{EXTRACT_PATH}/{name}'
        with open(f'{path}/beatmap.json', 'r') as f:
            beatmap = json.load(f)
        targets.append([beatmap['xs'], beatmap['ys']])

    # pad for training
    pad = max(len(coord[0]) for coord in targets)
    for i, t in enumerate(targets):
        for j in range(2):
            pad_len = pad-len(t[j])
            if pad_len != 0:
                t[j].extend([j]*(pad_len))
            t[j] = torch.tensor(t[j])
        targets[i] = torch.stack((t[0], t[1]))
    targets = torch.stack(targets)
    dataset = torch.full((len(targets), 2, pad), 0.0)
    dataset += targets
    return dataset

if __name__ == '__main__':
    extract()
