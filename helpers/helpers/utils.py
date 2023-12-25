#------------------------------------------------------------------------------#
#   utils.py is a python file containing all the utility helper functions 
#   we need in this project
#------------------------------------------------------------------------------#
from PyQt5.QtGui import QImage, qRgb
import numpy as np
import os
from pathlib import Path
import cv2
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import PIL.Image as Image
from skimage import color
import matplotlib.pyplot as plt
import pandas as pd
from PyQt5.QtGui import QPixmap
from itertools import cycle
import inspect

gray_color_table = [qRgb(i, i, i) for i in range(256)]

pharengeal_file_dir =  "data/subjects_data.csv"
stops_file_dir = "data/v-stops_all.csv" 
pharengeal_df = pd.read_csv(pharengeal_file_dir)
s_d_df        = pd.read_csv(stops_file_dir)








def frame_finder(subjects_dir , subject_name, gt_type="static_dynamic"):
    folder_name = "frames"
    non_pharyngeal = []
    participant = subject_name[0:-1].upper()
    # print(participant.upper())
    file = subject_name[-1].lower()
    # print(file)
    
    if gt_type == "static_dynamic":
        data = s_d_df.loc[(s_d_df["participant"] == participant.upper()) & (s_d_df["file_num"] == file)]
        print(data)
        for i in data.index:
            non_pharyngeal_parts = [f'{j}.jpeg'  for j in range(int(data["start"][i]), int(data["end"][i])+1)]
            non_pharyngeal.append(non_pharyngeal_parts)
        
        subject_pharyngeal = [item for sublist in non_pharyngeal for item in sublist]
        # print(subject_pharyngeal)
        subject_non_pharyngeal = sorted(list(set(os.listdir(os.path.join(subjects_dir, folder_name))) - set(subject_pharyngeal)))
        temp = sorted([int(item.split('.')[0]) for item in subject_non_pharyngeal])
        subject_non_pharyngeal = [f"{item}.jpeg" for item in temp]
        # print("=========================")
        # print(subject_non_pharyngeal)

        zip0 = list(zip(subject_pharyngeal, cycle('0')))   # [ (name.png , 0), (name2.png, 0), ...... ]
        zip1 = list(zip(subject_non_pharyngeal, cycle('1')))
        zipped = zip0 + zip1
        sorted_zipped = sorted(zipped, key = lambda x: int(x[0].split('.')[0]))

        print(sorted_zipped)
        subject_label = []     # list of zeros and ones [0, 0, 0, 0, 1, 1, 1, 1, ...]

        for i in range(len(sorted_zipped)):
            subject_label.append(int(sorted_zipped[i][1]))

        subject_frame_count = list(range(0, len(os.listdir(os.path.join(subjects_dir, folder_name)))))

    elif gt_type == "pharengeal":    
        data = pharengeal_df.loc[(pharengeal_df["participant"] == participant.upper()) & (pharengeal_df["file_num"] == file)]
        for i in data.index:
            non_pharyngeal_parts = [f'{j}.jpeg'  for j in range(int(data["start"][i]), int(data["end"][i])+1)]
            non_pharyngeal.append(non_pharyngeal_parts)
    
        subject_pharyngeal = [item for sublist in non_pharyngeal for item in sublist]
        subject_non_pharyngeal = sorted(list(set(os.listdir(os.path.join(subjects_dir, folder_name))) - set(subject_pharyngeal)))
        temp = sorted([int(item.split('.')[0]) for item in subject_non_pharyngeal])
        subject_non_pharyngeal = [f"{item}.jpeg" for item in temp]

        zip0 = list(zip(subject_pharyngeal, cycle('1')))   # [ (name.png , 0), (name2.png, 0), ...... ]
        zip1 = list(zip(subject_non_pharyngeal, cycle('0')))
        zipped = zip0 + zip1
        sorted_zipped = sorted(zipped, key = lambda x: int(x[0].split('.')[0]))
        subject_label = []     # list of zeros and ones [0, 0, 0, 0, 1, 1, 1, 1, ...]
    
        for i in range(len(sorted_zipped)):
            subject_label.append(int(sorted_zipped[i][1]))
    
        subject_frame_count = list(range(0, len(os.listdir(os.path.join(subjects_dir, folder_name)))))
        
    return subject_pharyngeal, subject_non_pharyngeal, subject_frame_count, subject_label[:-1]

def get_indices(arr):
    dynamic_frames_begin_pos = []
    dynamic_frames_end_pos = []
    for idx in range(1 ,len(arr)):
        prev = arr[idx - 1]
        curr = arr[idx]

        if curr == 1 and prev == 0:
            dynamic_frames_begin_pos.append(idx)

        if curr == 0 and prev == 1 and len(dynamic_frames_begin_pos) != 0:
            dynamic_frames_end_pos.append(idx)

    if len(dynamic_frames_begin_pos) != len(dynamic_frames_end_pos):
        dynamic_frames_end_pos.append(len(arr) - 1)

    return dynamic_frames_begin_pos, dynamic_frames_end_pos

def filer_arr(arr, window_size=20, thr=16):

    filter_window = sum([e for e in arr[:window_size]])
    for idx in range(len(arr)):
        if idx < window_size:
            pass
        filter_window = filter_window + arr[idx] - arr[idx-window_size]
        if filter_window > thr:
            arr[idx-window_size+1: idx + 1] = 1
            idx = idx + window_size - thr
    return arr

def remove_notches(begin_arr, end_arr, static_notch_thr, dynamic_notch_thr):
    # print(f"begin arr: {len(begin_arr)}\n end arr: {len(end_arr)}")
    df = pd.DataFrame({'begin' : begin_arr, 'end': end_arr})
    df['frames_between'] = np.abs(df['end'].sub(df['begin']))
    df = df.loc[df['frames_between'] > dynamic_notch_thr, ['begin', 'end']]
    df['begin_shifted'] = df.begin.shift(periods=-1, axis='index')
    df.fillna(200000, inplace=True)
    df['frames_ahead'] = np.abs(df['end'].sub(df['begin_shifted']))
    temp_df = df[['end', 'frames_ahead']].copy()
    temp_df.where(temp_df['frames_ahead'] > static_notch_thr, np.nan, inplace=True)
    temp_df.fillna(method='bfill', inplace=True)
    df['end'] = temp_df['end']
    df.drop(columns=['frames_ahead'], inplace=True)
    df.drop_duplicates(subset='end', keep='first', inplace=True)
    df = df.astype('int')

    return df['begin'].values, df['end'].values

def bin_sequence_to_df(sequence_arr, save_path):
    begin_arr, end_arr = get_indices(sequence_arr)
    regions_dataframe = pd.DataFrame({'begin' : begin_arr, 'end': end_arr})
    regions_dataframe.to_csv(save_path)

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def rotate_point(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + (np.cos(angle) * (px - ox)) - (np.sin(angle) * (py - oy))
    qy = oy + (np.sin(angle) * (px - ox)) + (np.cos(angle) * (py - oy))
    return [int(np.ceil(qx)), int(np.ceil(qy))]

def translate_point(point, d_x, d_y):
    x, y = point
    return [int(np.ceil(x+d_x)), int(np.ceil(y+d_y))]

def convert_normal_coor(box, img):
    h, w, d = img.shape
    ymin, xmin, ymax, xmax = box
    xmin, xmax = np.ceil(xmin*w), np.ceil(xmax*w)
    ymin, ymax = np.ceil(ymin*h), np.ceil(ymax*h)
    l = list(map(int, [xmin, ymin, xmax, ymax]))
    
    return l

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var and len(var_name) > 3]