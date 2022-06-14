"""Optional file reading for additional file types.

Requires pydub and all its dependencies (mainly ffmpeg/libav)

"""
import numpy as np
from os.path import splitext
from pydub import AudioSegment
from typing import Tuple


def file_to_ndarray(file_path: str) -> Tuple[int, np.ndarray, AudioSegment]:
    """Extract an ndarray from a file.

    Uses pydub to support loading any audio file that ffmpeg/libav supports.
    Supports Mono and Stereo files,
    If more channels exist, the first two will be returned.

    Parameters
    ----------
    file_path : str
        Path to the audio file.

    Returns
    -------
    audio_seg.frame_rate: int
        The sample rate of the signal.
    output_sig : np.ndarray
        The output signal.
    audio_seg : AudioSegment
        The AudioSegment created by pydub.

    """
    file_type = splitext(file_path)[1][1:]
    audio_seg = AudioSegment.from_file(file_path, file_type)
    output_sig = segment_to_ndarray(audio_seg)
    return audio_seg.frame_rate, output_sig, audio_seg


def segment_to_ndarray(audio_seg: AudioSegment) -> np.ndarray:
    """Given an AudioSegment, return PCM audio data as an nd numpy array."""
    samples = np.array(audio_seg.get_array_of_samples())
    if audio_seg.channels == 1:
        return samples
    L_channel, R_channel = samples[::2], samples[1::2]
    return np.column_stack((L_channel, R_channel))
