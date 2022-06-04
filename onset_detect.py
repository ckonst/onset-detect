# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 02:40:34 2020

@author: Christian Konstantinov
"""

import numpy as np
import librosa as lb
import torch

from extraction.extract_features import extract_features
from model.hyperparameters import DSP

def get_lmfs(fs, input_sig, W, stride, fmin=20.0, fmax=20000.0):
    """Return the log mel frequency spectrogram."""
    input_sig = input_sig.astype(np.float32)
    mfs = lb.feature.melspectrogram(input_sig, sr=fs, n_fft=W,
                                     hop_length=stride,
                                     fmin=fmin, fmax=fmax)
    lmfs = lb.power_to_db(mfs, ref=np.max)
    return lmfs

def superflux(fs, input_sig):
    """Return a vector of onset times for input_sig."""
    W = 1024
    stride = int(lb.time_to_samples(0.01, sr=fs))
    lag = 2
    max_size = 3
    lmfs = get_lmfs(fs, input_sig, W, stride)
    # use the log MFC for superflux onset detection
    # superflux function
    odf_sf = lb.onset.onset_strength(S=lmfs, sr=fs,
                                         hop_length=stride,
                                         lag=lag, max_size=max_size)
    onset_sf = lb.onset.onset_detect(onset_envelope=odf_sf,
                                      sr=fs,
                                      hop_length=stride,
                                      units='time')
    return onset_sf

def get_onset_frames(predictions: torch.Tensor):
    """
    Given a time-domain Onset Detection Function (ODF) as a torch Tensor,
    return the selected peaks as STFT frame indices.

    Parameters
    ----------
    predictions : torch.Tensor
        The predicted ODF.

    Returns
    -------
    p : np.ndarray
        The selected peaks as STFT frame indices.

    """
    p = predictions.cpu().detach().numpy()
    p = lb.util.peak_pick(p, 7, 7, 7, 7, 0.5, 5)
    p,= np.nonzero(p)
    return p

def get_onset_times(predictions: torch.Tensor, fs: int, stride: int):
    """
    Given a time-domain Onset Detection Function (ODF) as a torch Tensor,
    return the time-domain onsets as a numpy array.

    Parameters
    ----------
    predictions : torch.Tensor
        The predicted ODF.
    fs : int
        The sample rate.
    stride : int
        The stride for the STFT.

    Returns
    -------
    p : np.ndarray
        The selected time-domain onsets in seconds.

    """
    p = get_onset_frames(predictions)
    p = lb.frames_to_time(p, fs, stride)
    return p

# TODO: make an inference class to generalize to other applications.
def neural_onsets(audio_path, model_path, dsp: DSP):
    """
    Inference for the learned neural network model.

    Parameters
    ----------
    audio_path : str
        The path to the audio file.
    model_path : str
        The path to the neural network model.
    dsp : DSP
        DSP hyperparameters.

    Returns
    -------
    np.ndarray
        The selected time-domain onsets in seconds.

    """
    model = torch.load(model_path)
    (spectrogram, indices), targets = extract_features(audio_path, dsp)
    predictions = model((spectrogram, indices))
    return get_onset_times(predictions, dsp.fs, dsp.stride)

def create_click_track(onset_times, input_sig=None):
    """
    Create a click track of the given onsets.
    If input_sig is provided the click track will be added to the signal.

    Parameters
    ----------
    onset_times : np.ndarray
        The time-domain onsets in seconds.
    input_sig : np.ndarray, optional
        The audio for perceptual evaluation. The default is None.

    Returns
    -------
    click_track : np.ndarray
        The output click track signal.

    """
    click, _ = lb.load('./baseline/click.wav')
    if input_sig is None:
        click_track = np.zeros(onset_times[-1] + click.size)
    else:
        click_track = input_sig
    for o in onset_times:
        click_track[o:o+click[:, 0].size] += click[:click_track.size-o, 0]
    return click_track


