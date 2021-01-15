# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 02:40:34 2020

@author: Christian Konstantinov
"""

import numpy as np
import librosa as lb

def get_lmfs(fs, input_sig, W, stride, fmin=20.0, fmax=20000.0):
    """Return the log mel frequency spectrogram."""
    mfs = lb.feature.melspectrogram(input_sig, sr=fs, n_fft=W,
                                     hop_length=stride,
                                     fmin=fmin, fmax=fmax)
    lmfs = lb.power_to_db(mfs, ref=np.max)
    return lmfs

def superflux(fs, input_sig):
    """Return a vector of onset times for input_sig."""
    W = 1024
    stride = int(lb.time_to_samples(1./100, sr=fs))
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