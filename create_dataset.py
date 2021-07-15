# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 22:29:13 2021

@author: Christian Konstantinov
"""
import torch
from torch.utils.data import Dataset
from glob import glob
import json
import librosa as lb

DATA_PATH = './dataset/osu'
RAW_PATH = f'{DATA_PATH}/raw'
EXTRACT_PATH = f'{DATA_PATH}/extracted'
TENSOR_PATH = './dataset/tensors'

class OnsetDataset(Dataset):

    def __init__(self, features_file: str, targets_file: str, W: int):
        self.features = torch.load(features_file)
        with open(targets_file, 'r') as f:
            self.targets = json.load(f)

    def __len__(self):
        return len([_ for _ in glob(f'{RAW_PATH}/*/')])

    def __getitem__(self, index):
        pass

def create_onset_labels():
    for folder in glob(f'{RAW_PATH}/*/'):
        W = 1024
        fs = 8000
        stride = 80

        name = folder.split('raw\\')[1][:-1]
        path = f'{EXTRACT_PATH}/{name}'
        with open(f'{path}/beatmap.json', 'r') as f:
            beatmap = json.load(f)
        features = torch.load(f'{path}/features.pt')
        onsets = lb.time_to_frames(beatmap['onsets'], sr=fs, hop_length=stride, n_fft=W)
        targets = torch.zeros(features.shape[2])
        for o in onsets:
            targets[o] = 1
        torch.save(targets, f'{path}/targets.pt')


def create_coordinate_labels():
    targets = []
    for folder in glob(f'{EXTRACT_PATH}/*/'):
        name = folder.split('extracted\\')[1][:-1]
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
    torch.save(dataset, f'{TENSOR_PATH}/coordinate_data.pt')

#%%
if __name__ == '__main__':
    create_onset_labels()