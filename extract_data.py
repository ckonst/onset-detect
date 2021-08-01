# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 18:42:32 2021

@author: Christian Konstantinov
"""

from glob import glob
import json
import os

from file_reading import file_to_ndarray
from folder_iterator import iterate_folder

DATA_PATH = './dataset/osu'
RAW_PATH = f'{DATA_PATH}/raw'
EXTRACT_PATH = f'{DATA_PATH}/extracted'

def write_json(name):
    OSU_X = 640 # max Osupixel x value
    OSU_Y = 480 # max Osupixel y value
    map_path = f'{RAW_PATH}/{name}'
    folder_path = f'{EXTRACT_PATH}/{name}'
    json_file_path = f'{folder_path}/beatmap.json'

    out_data = {'name': name,'onsets': [], 'xs': [], 'ys': []}

    # read the raw data
    with open(glob(f'{map_path}/*.osu')[0], 'r', encoding='utf-8') as f:
        flag = False
        for line in f:
            if flag:
                data = line.split(',')
                out_data['onsets'].append(int(data[2])/1000)
                out_data['xs'].append(int(data[0])/OSU_X)
                out_data['ys'].append(int(data[1])/OSU_Y)
                continue
            if '[HitObjects]' in line:
                flag = True

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
