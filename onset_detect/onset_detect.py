import librosa as lb
import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.io.wavfile import write
from torch.utils.data import DataLoader, Dataset

from onset_detect.extraction.extract_features import extract_features_for_inference
from onset_detect.model.hyperparameters import DSP, ML
from onset_detect.model.model import OnsetDetector


def get_lmfs(
    fs: float,
    input_sig: NDArray,
    W: int,
    stride: int,
    fmin: float = 20.0,
    fmax: float = 20000.0,
) -> NDArray:
    """Return the log mel frequency spectrogram."""
    input_sig = input_sig.astype(np.float32)
    mfs = lb.feature.melspectrogram(
        input_sig, sr=fs, n_fft=W, hop_length=stride, fmin=fmin, fmax=fmax
    )
    lmfs = lb.power_to_db(mfs, ref=np.max)
    return lmfs


def superflux(fs: float, input_sig: NDArray) -> NDArray:
    """Return a vector of onset times for input_sig using the superflux method."""
    W = 1024
    stride = int(lb.time_to_samples(0.01, sr=fs))
    lag = 2
    max_size = 3
    lmfs = get_lmfs(fs, input_sig, W, stride)
    # use the log MFC for superflux onset detection
    # superflux function
    odf_sf = lb.onset.onset_strength(
        S=lmfs, sr=fs, hop_length=stride, lag=lag, max_size=max_size
    )
    onset_sf = lb.onset.onset_detect(
        onset_envelope=odf_sf, sr=fs, hop_length=stride, units='time'
    )
    return onset_sf


def inos2(input_sig: np.ndarray, dsp: DSP, gamma: float = 95.5) -> torch.Tensor:
    """Return an ODF from input_sig using the INOS² method."""

    stft = lb.stft(input_sig, n_fft=dsp.W, hop_length=dsp.stride)
    magnitude = np.abs(stft)
    log_magnitude = np.log(magnitude + 1e-10)[
        1:-1, :
    ]  # remove DC and Nyquist components

    # number bins to keep based on gamma percentile of lowest magnitudes
    num_bins = np.floor((gamma / 100) * log_magnitude.shape[0]).astype(int)
    # sort each frequency bin's magnitudes and keep the lowest energy bins
    low_energy_bins = np.sort(log_magnitude, axis=0)[:num_bins, :]

    l2_norm_squared = np.linalg.norm(low_energy_bins, ord=2, axis=0) ** 2
    l4_norm = np.linalg.norm(low_energy_bins, ord=4, axis=0)

    inos2_odf = l2_norm_squared / l4_norm
    inos2_odf /= np.max(inos2_odf)  # normalize

    return inos2_odf


def ninos2(input_sig: np.ndarray, dsp: DSP, gamma: float = 95.5) -> torch.Tensor:
    """Return an ODF from input_sig using the NINOS² method."""

    stft = lb.stft(input_sig, n_fft=dsp.W, hop_length=dsp.stride)
    magnitude = np.abs(stft)
    log_magnitude = np.log(magnitude + 1e-10)[
        1:-1, :
    ]  # remove DC and Nyquist components

    # number bins to keep based on gamma percentile of lowest magnitudes
    num_bins = np.floor((gamma / 100) * log_magnitude.shape[0]).astype(int)
    # sort each frequency bin's magnitudes and keep the lowest energy bins
    low_energy_bins = np.sort(log_magnitude, axis=0)[:num_bins, :]

    l2_norm = np.linalg.norm(low_energy_bins, ord=2, axis=0)
    l4_norm = np.linalg.norm(low_energy_bins, ord=4, axis=0)
    sparsity_factor_denominator = num_bins ** (1 / 4.0) - 1  # L4 norm exponent

    ninos2_odf = (l2_norm / sparsity_factor_denominator) * (l2_norm / l4_norm - 1)
    ninos2_odf /= np.max(ninos2_odf)  # normalize

    return ninos2_odf


def get_onset_frames(predictions: torch.Tensor | NDArray) -> NDArray:
    """Return the selected peaks as STFT frame indices."""
    p = (
        predictions.cpu().detach().numpy()
        if isinstance(predictions, torch.Tensor)
        else predictions
    )
    p = lb.util.peak_pick(
        p, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10
    )
    (p,) = np.nonzero(p)
    return p


def get_onset_samples(predictions: torch.Tensor | NDArray, stride: int) -> NDArray:
    """Return the time-domain onsets as a numpy array."""
    p = get_onset_frames(predictions)
    p = lb.frames_to_samples(p, hop_length=stride)
    return p


def get_onset_times(
    predictions: torch.Tensor | NDArray, fs: int, stride: int
) -> NDArray:
    """Return the time-domain onsets as a numpy array."""
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


def neural_onsets(
    audio_path: str,
    model_path: str,
    dsp: DSP = None,
    ml: ML = None,
    device: torch.device = None,
    units='seconds',
) -> tuple[NDArray, torch.Tensor]:
    """Inference for the learned neural network model."""

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


def create_click_track(
    onset_times: NDArray, input_sig: NDArray | None = None
) -> NDArray:
    """Create a click track of the given onsets. If input_sig is provided the click track will be added to the signal."""
    click, _ = lb.load('./audio/click.wav')
    if input_sig is None:
        click_track = np.zeros(onset_times[-1] + click.size)
    else:
        click_track = input_sig
    for o in onset_times:
        click_track[o : o + click.size] += click[: click_track.size - o]
    return click_track


def main() -> None:
    dsp = DSP(fs=44100)
    # ml = ML()
    # onset_times, predictions = neural_onsets(
    #     './audio/pop_shuffle.wav',
    #     './onset_detect/model/trained_models/f_0.13303186715466744.pt',
    #     dsp=dsp,
    #     ml=ml,
    #     units='samples',
    # )
    audio_sig, _ = lb.load('./audio/pop_shuffle.wav', sr=dsp.fs)
    audio_sig /= np.max(np.abs(audio_sig))

    ninos2_odf = ninos2(audio_sig, dsp, gamma=50)

    ninos2_onsets = lb.frames_to_samples(
        lb.onset.onset_detect(
            onset_envelope=ninos2_odf,
            sr=dsp.fs,
            hop_length=dsp.stride,
            units='frames',
        ),
        hop_length=dsp.stride,
    )

    fig, ax = plt.subplots()
    ax.set_title('NINOS² Onsets')
    ax.vlines(ninos2_onsets, 0, 1, label='Onsets')
    plt.show()

    fig, ax = plt.subplots()
    ax.set_title('NINOS² Onset Detection Function')
    ax.plot(ninos2_odf, label='NINOS² ODF')
    plt.show()

    click_track = create_click_track(ninos2_onsets, input_sig=audio_sig)
    write('./audio/pop_shuffle_onsets.wav', dsp.fs, click_track)


if __name__ == '__main__':
    main()
