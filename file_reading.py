import numpy as np
from pydub import AudioSegment

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