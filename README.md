# onset_detect
A package for polyphonic music onset detection models for automatic rhythm game beatmap creation.
### Data
The data is composed of beatmaps from the rhythm game, *Osu!*

Each beatmap is contained in a folder named after the song and has 2 files: a .mp3 and a .osu

.osu files are human readable specifications of the beatmap data, including onset times and circle locations.

This package only uses the latter two datapoints for simplicity and use in [prototypes](https://github.com/ckonst/Impulse).

### Setup

1. Requires torch, torchaudio, librosa, mir-eval and all their dependencies

    - `pip install requirements.txt`

2. The data can be downloaded [here](https://www.dropbox.com/sh/sxbkcq7ulnmpdx1/AADaMk0guIlP87fsQIPemmSxa?dl=0)
    - Put the `osu/` folder in `dataset/` and run `python -m extraction.extract_data`.

    - This will create a similar folder structure under `dataset/extracted/`.

3. Run `python -m extraction.extract_features` to extract log mel spectrograms from the mp3s
    - Each folder now has tensors saved as .pt files

4. Now you can run train the model with `python -m model.train`. 
    - The best scoring models will be saved in `model/trained_models/`.

### Model

The baseline model in `baseline/` uses pre-emphasis and low-pass filtering, and half-wave rectification to create an onset dectection function.

The model defined in `model/model.py` uses 3 CNN layers, each with batch normalization and ReLU activations,
followed by a BLSTM layer, 50% dropout, 6 CNN layers, each with batch normalization, ReLU, and Maxpooling in the frequency dimension,
following by 3 fully connected layers, each with a 50% dropout layer.