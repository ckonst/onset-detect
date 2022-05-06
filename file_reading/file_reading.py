import numpy as np
from os.path import splitext
from pydub import AudioSegment
from typing import Tuple

def file_to_ndarray(file_path: str) -> Tuple[int, np.ndarray, AudioSegment]:
    """
    Extract an ndarray from file.

    Given a path to a file,
    return a tuple conataining the sample rate,
    an nd numpy array containing PCM audio data,
    and an AudioSegment in audio_seg.
    """
    file_type = splitext(file_path)[1][1:]
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