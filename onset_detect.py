import numpy as np
import librosa as lb
import torch

from scipy.io.wavfile import write
from torch.utils.data import DataLoader, Dataset

from extraction.extract_features import extract_features_for_inference
from model.hyperparameters import DSP, ML
from model.model import OnsetDetector

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
    """Return the selected peaks as STFT frame indices.

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
    p = lb.util.peak_pick(p, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
    p,= np.nonzero(p)
    return p

def get_onset_samples(predictions: torch.Tensor, stride: int) -> np.ndarray:
    """Return the time-domain onsets as a numpy array.

    Parameters
    ----------
    predictions : torch.Tensor
        The predicted ODF.
    stride : int
        The stride for the STFT.

    Returns
    -------
    p : np.ndarray
        The selected time-domain onsets in samples.

    """
    p = get_onset_frames(predictions)
    p = lb.frames_to_samples(p, hop_length=stride)
    return p

def get_onset_times(predictions: torch.Tensor, fs: int, stride: int) -> np.ndarray:
    """Return the time-domain onsets as a numpy array.

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
    p = lb.frames_to_time(p, sr=fs, hop_length=stride)
    return p

class Inference(Dataset):
    def __init__(self, tensor, indices):
        self.tensor = tensor
        self.indices = indices
        self._size = len(indices)
    def __len__(self):
        return self._size
    def __getitem__(self, index: int):
        frame = index
        tensor, indices = self.tensor, self.indices
        context = tensor.shape[-1] // len(indices)
        start = frame * context
        end = (frame + 1) * context
        frame = torch.FloatTensor([frame])
        return tensor[:, :, start:end], frame

# TODO: add tests
# TODO: debug, maybe plot predictions
def neural_onsets(audio_path, model_path, dsp: DSP = None, ml: ML = None, device: torch.device = None, units='seconds'):
    """Inference for the learned neural network model.

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
        The selected time-domain onsets with the selected units.

    """
    if dsp is None:
        dsp = DSP()
    if ml is None:
        ml = ML
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OnsetDetector(**ml.__dict__, device=device).to(device)
    model.load_state_dict(torch.load(model_path))
    tensor, indices = extract_features_for_inference(audio_path, dsp)
    inference_dataset = Inference(tensor, indices)
    loader = DataLoader(inference_dataset, batch_size=ml.batch_size, pin_memory=True)
    output_tensors = []
    with torch.no_grad():
        for tensor, indices in loader:
            tensor = tensor.to(device)
            indices = indices.to(device)
            predictions = model((tensor, indices))
            output_tensors.append(torch.flatten(predictions))

    predictions = torch.cat(output_tensors)

    if units == 'samples':
        onsets = get_onset_samples(predictions, dsp.stride)
    elif units == 'seconds':
        onsets = get_onset_times(predictions, dsp.fs, dsp.stride)
    elif units == 'frames':
        onsets = get_onset_frames(predictions)
    else:
        raise ValueError('Invalid unit type.')

    return onsets, predictions

def create_click_track(onset_times, input_sig=None):
    """Create a click track of the given onsets.

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
    click, _ = lb.load('./audio/click.wav')
    if input_sig is None:
        click_track = np.zeros(onset_times[-1] + click.size)
    else:
        click_track = input_sig
    for o in onset_times:
        click_track[o:o+click.size] += click[:click_track.size-o]
    return click_track

def main():
    dsp = DSP()
    ml = ML()
    onset_times, predictions = neural_onsets('./audio/pop_shuffle.wav',
                                './model/trained_models/f_0.13303186715466744.pt', dsp=dsp, ml=ml, units='samples')
    audio_sig, fs = lb.load('./audio/pop_shuffle.wav', sr=dsp.fs)
    click_track = create_click_track(onset_times, input_sig=audio_sig)
    write('./audio/pop_shuffle_onsets.wav', dsp.fs, click_track)

if __name__ == '__main__':
    main()