# onset-detect
A package for polyphonic music onset detection models for automatic rhythm game beatmap creation.
### Data
The data is composed of beatmaps from the rhythm game, *Osu!*

Each beatmap is contained in a folder named after the song and has 2 files: a .mp3 and a .osu

.osu files are human readable specifications of the beatmap data, including onset times and circle locations.

This package only uses the latter two datapoints for simplicity and use in prototypes.

### Setup

1. Setup ffmpeg/libav according to the [pydub setup guide](https://github.com/jiaaro/pydub?tab=readme-ov-file#getting-ffmpeg-set-up)
2. Install uv, if you haven't already. In the root folder of the repo run:

    - `uv init`

2. The data can be downloaded [here](https://www.dropbox.com/sh/sxbkcq7ulnmpdx1/AADaMk0guIlP87fsQIPemmSxa?dl=0)
    - Put the `osu/` folder in `dataset/` and run `uv run python -m onset_detect.extraction.extract_data`.

    - This will create a similar folder structure under `onset_detect/dataset/extracted/`.

3. Run `uv run python -m onset_detect.extraction.extract_features` to extract log mel spectrograms from the mp3s
    - Each folder now has tensors saved as .pt files

4. Now you can run train the model with `uv run python -m onset_detect.model.train`. 
    - The best scoring models will be saved in `onset_detect/model/trained_models/`.
