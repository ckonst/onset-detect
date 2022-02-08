# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 18:42:32 2021

@author: Christian Konstantinov
"""

from glob import glob
from math import cos, pi

import json
import os

from file_reading.folder_iterator import iterate_folder

DATA_PATH = '../dataset/osu'
RAW_PATH = f'{DATA_PATH}/raw'
EXTRACT_PATH = f'{DATA_PATH}/extracted'

def write_json(name):
    # Constants defined by the .osu file format:
    # https://osu.ppy.sh/wiki/en/Client/File_formats/Osu_%28file_format%29
    OSU_X = 640 # max Osupixel x value.
    OSU_Y = 480 # max Osupixel y value.
    map_path = f'{RAW_PATH}/{name}'
    folder_path = f'{EXTRACT_PATH}/{name}'
    json_file_path = f'{folder_path}/beatmap.json'

    out_data = {'name': name,'onsets': [], 'xs': [], 'ys': []}

    # read the raw data
    with open(glob(f'{map_path}/*.osu')[0], 'r', encoding='utf-8') as f:
        hit_objects_flag = False
        timing_points_flag = False

        timing_points = []
        timing_pointer = 0
        slider_multiplier = 1

        for line in f:
            data = line.strip().split(',')
            if hit_objects_flag:
                onset = int(data[2]) / 1000
                type_ = int(data[3])

                # Inital onset for all Hit Object types.
                out_data['onsets'].append(onset)
                out_data['xs'].append(int(data[0]) / OSU_X)
                out_data['ys'].append(int(data[1]) / OSU_Y)

                if onset > timing_points[timing_pointer][0] and timing_pointer < len(timing_points) - 1:
                    timing_pointer += 1

                # Slider onsets past the inital onset must be calculated manually.
                if type_ == 2 or type_ == 6:
                    slides = int(data[6])
                    length = float(data[7])
                    beat_length = float(timing_points[timing_pointer][1])

                    # If the beat length is inherited, it must be converted back to a usable value.
                    if not int(timing_points[timing_pointer][2]):
                        beat_length = 100 * (1 / -beat_length)
                    # This formula is taken directly from Osu's wiki.
                    dur = length / (slider_multiplier * beat_length * 100)
                    out_data['onsets'].extend([onset + dur * i for i in range(1, slides+1)])

                    # Compute direction of the next onset by averaging the curve_points of the slider.
                    x0, y0 = int(data[0]), int(data[1])
                    curve_points = data[5].split('|')
                    cps = list(map(lambda x: x.split(':'), curve_points))[1:]
                    x1 = sum([int(x) for x, _ in cps]) / len(list(cps))
                    y1 = sum([int(y) for _, y in cps]) / len(list(cps))
                    sx, sy = min(max(x1-x0, -1), 1), min(max(y1-y0, -1), 1) # clamp values
                    # Cos(pi*t) will map to {1, -1}, and oscillate as t increases monotonically.
                    # This will change the onset location according to the slider's behavior in Osu.
                    out_data['xs'].extend([(sx*cos(pi*t)*length + x0) / OSU_X for t in range(slides)])
                    out_data['ys'].extend([(sy*cos(pi*t)*length + y0) / OSU_Y for t in range(slides)])

                continue

            if timing_points_flag:
                if not data[0]:
                    timing_points_flag = False
                    continue
                # (Time (ms), Multiplier, uninherited)
                timing_points.append((float(data[0]), float(data[1]), int(data[6])))

            if '[HitObjects]' in line:
                hit_objects_flag = True
                continue
            if '[TimingPoints]' in line:
                timing_points_flag = True
                continue
            if 'SliderMultiplier' in line:
                slider_multiplier = float(line.strip().split(':')[-1])

    # write to json
    with open(json_file_path, 'w+') as f:
        json.dump(out_data, f)

def extract():
    """Extract all the data from the osu folder."""
    for name in iterate_folder(RAW_PATH):
        path = f'{EXTRACT_PATH}/{name}'
        if not os.path.exists(path):
            os.makedirs(path)
        write_json(name)
#%%
if __name__ == '__main__':
    extract()
