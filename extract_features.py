# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 14:35:39 2021

@author: Christian Konstantinov
"""
import torch
from torchaudio import transforms
import os
from glob import glob
from pydub import AudioSegment
import numpy as np
import librosa as lb

DATA_PATH = './dataset/osu'
RAW_PATH = f'{DATA_PATH}/raw'
EXTRACT_PATH = f'{DATA_PATH}/extracted'

def extract(W: int = 1024) -> None:
    """Extract all the data from the osu folder."""
    for folder in glob(f'{RAW_PATH}/*/'):
        name = folder.split('raw\\')[1][:-1]
        path = f'{EXTRACT_PATH}/{name}'
        FOLDER_PATH = f'{DATA_PATH}/extracted/{name}'
        if not os.path.exists(path):
            os.makedirs(path)
        #lmfs = torch.cat((get_lmfs(name, W=1024), get_lmfs(name, W=2048), get_lmfs(name, W=4096)), dim=0)
        lmfs = get_lmfs(name, W, 8000)
        lmfs = pad_tensor(lmfs, lmfs.shape[2], W)
        torch.save(lmfs, f'{FOLDER_PATH}/features.pt')

def pad_tensor(unpadded: torch.Tensor, size: int, W: int) -> torch.Tensor:
    pad_sig = torch.zeros(unpadded.shape[0], unpadded.shape[1], size + (W - (size % W)))
    pad_sig[:, :, :size] = unpadded
    return pad_sig

def get_lmfs(name: str, W: int, sr: int, f_min: float = 20.0, f_max: float = 4000.0) -> torch.Tensor:
    """Return the log mel frequency spectrogram."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MAP_PATH = f'{RAW_PATH}/{name}'
    # read the mp3 data
    fs, stereo_sig, _ = file_to_ndarray(glob(f'{MAP_PATH}/*.mp3')[0], 'mp3')
    stride = int(sr*0.01)
    stereo_sig = stereo_sig.astype(np.float32) # convert to floating point
    stereo_sig = lb.resample(stereo_sig.T, fs, sr, res_type='kaiser_fast', fix=True)
    stereo_sig = torch.from_numpy(stereo_sig).to(device) # reshape and send to cuda, if available
    mono_sig = torch.mean(stereo_sig, dim=0, keepdim=True) # convert to mono
    norm_sig = normalize(mono_sig)
    # get the mel-frequency spectrum
    mfs = transforms.MelSpectrogram(sample_rate=fs, n_fft=W,
                                    f_min=f_min, f_max=f_max,
                                    n_mels=80, hop_length=stride,
                                    window_fn=torch.hamming_window).to(device)(norm_sig)
    lmfs = transforms.AmplitudeToDB().to(device)(mfs) # convert to log-mfs
    lmfs = normalize(lmfs)
    return lmfs

def file_to_ndarray(file_path: str, file_type: str) -> (int, np.ndarray, AudioSegment):
    """Given a path to a file, and its type (extension without '.'),
       Return a tuple conataining
       the sample rate, an nd numpy array containing PCM audio data,
       and an AudioSegment in audio_seg.
    """
    audio_seg = AudioSegment.from_file(file_path, file_type)
    output_sig = segment_to_ndarray(audio_seg)
    return audio_seg.frame_rate, output_sig, audio_seg

def segment_to_ndarray(audio_seg: AudioSegment) -> np.ndarray:
    """Given an AudioSegment, return a nd numpy array containing PCM audio data."""
    samples = np.array(audio_seg.get_array_of_samples())
    if audio_seg.channels == 1:
        return samples
    L_channel, R_channel = samples[::2], samples[1::2]
    return np.column_stack((L_channel, R_channel))

def normalize(tensor: torch.Tensor) -> torch.Tensor:
    # Subtract the mean, and scale to the interval [-1, 1]
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean/tensor_minusmean.abs().max()

#%%

if __name__ == '__main__':
    extract()
    '''
    import matplotlib.pyplot as plt
    lmfs = get_lmfs('63112 ONE OK ROCK - Answer is Near', 1024, 8000).numpy()
    print(lmfs.shape)
    plt.figure()
    plt.imshow(lmfs[0,:,:1024])'''