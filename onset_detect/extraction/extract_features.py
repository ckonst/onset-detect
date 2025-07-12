import json
import logging
from glob import glob

import librosa as lb
import numpy as np
import torch
from numpy.typing import NDArray
from torchaudio import transforms

from onset_detect.evaluation.time_elapsed import timed
from onset_detect.file_reading.file_reading import file_to_ndarray
from onset_detect.file_reading.folder_iterator import subdir_names
from onset_detect.model.hyperparameters import DSP

log = logging.getLogger(__name__)

DATA_PATH = './onset_detect/dataset/osu'
RAW_PATH = f'{DATA_PATH}/raw'
EXTRACT_PATH = f'{DATA_PATH}/extracted'

# TODO:
# - Add log mel / STFT autocorrelation = |G(x)|² preprocessing
# - Multiple spectrogram feature processing
# - Maybe tempo estimation


@timed(units='s')
def extract() -> None:
    """Create processes to parallelize feature extraction."""
    for name in subdir_names(RAW_PATH):
        log.info('processing audio inside: %s', name)
        process_and_save_song(name)


def process_and_save_song(name: str) -> None:
    """Extract features from a given song name, then save them to the new folder."""
    (tensor, indices), targets = process_song(name)
    song_path = f'{EXTRACT_PATH}/{name}'
    log.info('Serializing features and targets')
    torch.save((tensor, indices), f'{song_path}/features.pt')
    torch.save(targets, f'{song_path}/targets.pt')


def process_song(name: str) -> tuple[tuple[torch.Tensor, int], torch.Tensor]:
    """Process one song and return target and feature Tensors with the index to the spectrogram frame."""
    dsp = DSP()
    song_path = f'{EXTRACT_PATH}/{name}'
    tensor, indices = extract_features(name, dsp)
    log.info('Creating onset labels')
    targets = create_onset_labels(tensor, song_path, dsp)
    return (tensor, indices), targets


def extract_features(name: str, dsp: DSP) -> torch.Tensor:
    """Extract the spectrogams and tensor indices for the dataset."""
    _, audio_sig = load_audio(name, target_fs=dsp.fs, mode='dir')
    log.info('Audio loaded, starting feature extraction')
    tensor = get_lmfs(audio_sig, dsp)
    log.info('Padding Mel Frequency Spectrogram')
    tensor = pad_tensor(tensor, tensor.shape[-1], dsp.context)
    log.info('Generating Indices')
    indices = list(range(0, tensor.shape[-1] // dsp.context))
    return tensor, indices


def extract_features_for_inference(path: str, dsp: DSP):
    _, audio_sig = load_audio(path, target_fs=dsp.fs, mode='file')
    tensor = get_lmfs(audio_sig, dsp)
    indices = list(range(0, tensor.shape[-1] // dsp.context))
    return tensor, indices


def create_onset_labels(
    features: torch.Tensor, song_path: str, dsp: DSP
) -> torch.Tensor:
    """Given the features (spectrogram) of the data, return the target onsets."""
    with open(f'{song_path}/beatmap.json', 'r') as f:
        beatmap = json.load(f)
    onsets = lb.time_to_frames(beatmap['onsets'], sr=dsp.fs, hop_length=dsp.stride)
    targets = torch.zeros(features.shape[2])
    for o in onsets:
        targets[o] = 1
    return targets


def load_audio(path: str, target_fs: int = 44100, mode='file') -> tuple[int, NDArray]:
    source_fs, audio = None, None
    if mode == 'file':
        source_fs, audio, _ = file_to_ndarray(path, as_mono=True)
    elif mode == 'dir':
        # use glob because We don't actually know the name of the mp3,
        # as it's not guarenteed to be the name of the directory
        (map_path,) = glob(f'{RAW_PATH}/{path}/*.mp3')
        source_fs, audio, _ = file_to_ndarray(map_path, as_mono=True)
        log.info('Loaded audio from path %s', map_path)
    if source_fs != target_fs:
        audio = lb.resample(audio, orig_sr=source_fs, target_sr=target_fs)
    return target_fs, audio


# FIXME: Long audio files use excessive RAM if sample rate is high (44.1kHz, etc.). Needs automatic processing in chunks.
# For now, keep sample rates low i.e. 8kHz maaaybe 16kHz
def get_lmfs(input_sig: np.ndarray, dsp: DSP) -> torch.Tensor:
    """Return the log mel frequency spectrogram."""
    mono_sig = torch.from_numpy(input_sig)
    norm_sig = normalize(mono_sig)
    log.info('Calculating Mel Frequency Spectrum')
    mfs = transforms.MelSpectrogram(
        sample_rate=dsp.fs,
        n_fft=dsp.W,
        f_min=dsp.f_min,
        f_max=dsp.f_max,
        n_mels=dsp.bands,
        hop_length=dsp.stride,
        window_fn=torch.hamming_window,
    )(norm_sig)
    log.info('Calculating Log Mel Frequency Spectrum')
    lmfs = transforms.AmplitudeToDB()(mfs).unsqueeze(0).detach()
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
    for name in subdir_names(EXTRACT_PATH):
        path = f'{EXTRACT_PATH}/{name}'
        with open(f'{path}/beatmap.json', 'r') as f:
            beatmap = json.load(f)
        targets.append([beatmap['xs'], beatmap['ys']])

    # pad for training
    pad = max(len(coord[0]) for coord in targets)
    for i, t in enumerate(targets):
        for j in range(2):
            pad_len = pad - len(t[j])
            if pad_len != 0:
                t[j].extend([j] * (pad_len))
            t[j] = torch.tensor(t[j])
        targets[i] = torch.stack((t[0], t[1]))
    targets = torch.stack(targets)
    dataset = torch.full((len(targets), 2, pad), 0.0)
    dataset += targets
    return dataset


# %%
if __name__ == '__main__':
    extract()
