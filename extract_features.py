# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 14:35:39 2021

@author: Christian Konstantinov
"""
import torch
import torchaudio
import os
from glob import glob
from pydub import AudioSegment
import numpy as np

DATA_PATH = './dataset/osu'
RAW_PATH = f'{DATA_PATH}/raw'
EXTRACT_PATH = f'{DATA_PATH}/extracted'

def extract():
    """Extract all the data from the osu folder."""
    for dir in glob(f'{RAW_PATH}/*/'):
        name = dir.split('raw\\')[1][:-1]
        path = f'{EXTRACT_PATH}/{name}'
        FOLDER_PATH = f'{DATA_PATH}/extracted/{name}'
        if not os.path.exists(path):
            os.makedirs(path)
        lmfs = torch.cat((get_lmfs(name, W=1024), get_lmfs(name, W=2048), get_lmfs(name, W=4096)), dim=0)
        torch.save(lmfs, f'{FOLDER_PATH}/features.pt')

def get_lmfs(name, W=1024, stride=441, f_min=20.0, f_max=20000.0):
    """Return the log mel frequency spectrogram."""
    MAP_PATH = f'{RAW_PATH}/{name}'

    # read the mp3 data
    fs, stereo_sig, _ = file_to_ndarray(glob(f'{MAP_PATH}/*.mp3')[0], 'mp3')
    stereo_sig = stereo_sig.astype(np.float32) # convert to floating point
    stereo_sig = torch.from_numpy(stereo_sig).T #reshape
    mono_sig = torch.mean(stereo_sig, dim=0, keepdim=True) # convert to mono
    # get the mel-frequency spectrum
    mfs = torchaudio.transforms.MelSpectrogram(sample_rate=fs, n_fft=W,
                                                f_min=f_min, f_max=f_max,
                                                n_mels=80, hop_length=stride,
                                                window_fn=torch.hamming_window)(mono_sig)
    return torchaudio.transforms.AmplitudeToDB()(mfs)

def file_to_ndarray(file_path, file_type):
    """Given a path to a file, and its type (extension without '.'),
       Return a tuple conataining
       the sample rate, an nd numpy array containing PCM audio data,
       and an AudioSegment in audio_seg.
    """
    audio_seg = AudioSegment.from_file(file_path, file_type)
    output_sig = segment_to_ndarray(audio_seg)
    return audio_seg.frame_rate, output_sig, audio_seg

def segment_to_ndarray(audio_seg):
    """Given an AudioSegment, return a nd numpy array containing PCM audio data."""
    samples = np.array(audio_seg.get_array_of_samples())
    if audio_seg.channels == 1:
        return samples
    L_channel, R_channel = samples[::2], samples[1::2]
    return np.column_stack((L_channel, R_channel))

#%%

if __name__ == '__main__':

    extract()
    '''
    import matplotlib.pyplot as plt
    lmfs = get_lmfs('63112 ONE OK ROCK - Answer is Near')
    plt.figure()
    plt.imshow(lmfs[0,:,:1024].numpy(), cmap='gray')
    '''