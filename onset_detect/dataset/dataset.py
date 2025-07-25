import random
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset, Subset

from onset_detect.file_reading.folder_iterator import subdir_names

DATA_PATH = './onset_detect/dataset/osu'
EXTRACT_PATH = f'{DATA_PATH}/extracted'


def _get_index_to_context() -> Dict[int, Tuple[str, int]]:
    """Create a mapping from Dataset index to name and context frame index.

    Returns
    -------
    index_to_context: Dict[int, Tuple[str, int]]
        A mapping from dataset index keys to name and context frame index pairs.

    """
    index_to_context = {}
    dataset_index = 0
    for name in subdir_names(EXTRACT_PATH):
        _, frame_indices = torch.load(f'{EXTRACT_PATH}/{name}/features.pt')
        for i in frame_indices:
            index_to_context[dataset_index + i] = (name, i)
        dataset_index += len(frame_indices)
    return index_to_context


def _get_name_to_index() -> Dict[str, Tuple[int, int]]:
    """Create a mapping from name to Dataset index and context frame.

    Returns
    -------
    name_to_index: Dict[str, Tuple[int, int]]
        A mapping from song names to Dataset index and context frame index pairs.

    """
    name_to_index = {}
    dataset_index = 0
    for name in subdir_names(EXTRACT_PATH):
        _, frame_indices = torch.load(f'{EXTRACT_PATH}/{name}/features.pt')
        for _ in frame_indices:
            name_to_index[name] = (dataset_index, dataset_index + len(frame_indices))
        dataset_index += len(frame_indices)
    return name_to_index


def _get_song_index_to_range(seed: int = 0) -> Dict[int, Tuple[int, int]]:
    song_index_to_range = {}
    dataset_index = 0
    songs = [name for name in subdir_names(EXTRACT_PATH)]
    random.seed(seed)
    random.shuffle(songs)
    for i, name in enumerate(songs):
        _, frame_indices = torch.load(f'{EXTRACT_PATH}/{name}/features.pt')
        song_index_to_range[i] = dataset_index, dataset_index + len(frame_indices) - 1
        dataset_index += len(frame_indices)
    return song_index_to_range


def _get_size() -> int:
    """Get the total size of the dataset in number of tensor partitions.

    Calculated using the list of indices for each partition.
    """
    return sum(
        [
            len(torch.load(f'{EXTRACT_PATH}/{name}/features.pt')[1])
            for name in subdir_names(EXTRACT_PATH)
        ]
    )


def _get_num_songs() -> int:
    """Get the number of songs in the dataset."""
    return len([0 for name in subdir_names(EXTRACT_PATH)])


class OnsetDataset(Dataset):
    """Dataset class for mapping spectrogram frames to onset classes."""

    def __init__(self, **kwargs):
        """Initialize the dataset with conversion maps, and loaded tensors."""
        self.__dict__.update(**kwargs)
        self.num_songs = _get_num_songs()
        self._size = _get_size()
        self.index_to_context = _get_index_to_context()
        self.name_to_index = _get_name_to_index()
        self.features, self.targets = self._load_tensors()

    def __len__(self) -> int:
        """Get the total size of the dataset in number of tensor partitions."""
        return self._size

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return a partition of the song's spectrogram, its index, and the ground truth sequence."""
        name, frame = self.index_to_context[index]
        tensor, indices = self.features[name]
        targets = self.targets[name]
        # tensor, indices = torch.load(f'{EXTRACT_PATH}/{name}/features.pt')
        # targets = torch.load(f'{EXTRACT_PATH}/{name}/targets.pt')
        context = tensor.shape[-1] // len(indices)
        start = frame * context
        end = (frame + 1) * context
        frame = torch.HalfTensor([frame])
        return ((tensor[:, :, start:end], frame), targets[start:end])

    def _load_tensors(self):
        features = {}
        targets = {}
        for name in subdir_names(EXTRACT_PATH):
            features[name] = torch.load(f'{EXTRACT_PATH}/{name}/features.pt')
            targets[name] = torch.load(f'{EXTRACT_PATH}/{name}/targets.pt')
        return features, targets

    def split_train_test_dev(self, seed: int = 0) -> Tuple[Subset, Subset, Subset]:
        """Return the dataset as subsets for training, testing, and validation.

        The data is split 80/10/10 by the number of songs in the dataset
        as opposed to the actual spectrogram frames.
        While this might make the code slightly more complicated,
        each frame should not be treated as an independent data point,
        they should instead form a coherent beatmap of the entire song
        (i.e. spectrogram partitions of the same song should be highly correlated).

        Parameters
        ----------
        seed : int, optional
            The seed for the shuffling of the songs. The default is 0.

        Returns
        -------
        Tuple[Subset, Subset, Subset]
            Training, Testing, Validation sets as a subset of this dataset.

        """
        dataset = []
        songs = []

        for i, name in enumerate(subdir_names(EXTRACT_PATH)):
            dataset += list(range(*self.name_to_index[name]))
            songs += [i]
        song_index_to_range = _get_song_index_to_range(seed)

        def map_fn(x):
            return song_index_to_range[int(round(x))][1]

        split = tuple(map(map_fn, (self.num_songs * 0.8, self.num_songs * 0.9)))
        return (
            Subset(self, dataset[0 : split[0]]),
            Subset(self, dataset[split[0] : split[1]]),
            Subset(self, dataset[split[1] :]),
        )


class CoordinateDataset(Dataset):
    """Dataset class for mapping Onset times to x, y coordinates."""

    # TODO: Finish implementing this class

    def __init__(self, **kwargs):
        """Write a useful docstring here."""
        self.__dict__.update(kwargs)

    def __len__(self):
        """Write a useful docstring here."""
        pass

    def __getitem__(self, index):
        """Write a useful docstring here."""
        pass
