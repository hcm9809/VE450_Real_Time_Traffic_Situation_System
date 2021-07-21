import pandas as pd
import numpy as np
from time import time
from math import atan2
import argparse
import os
import math

COCO_NAMES = {1: 'bicycle',
                2: 'car',
                3: 'motorbike',
                4: 'aeroplane',
                5: 'bus',
                6: 'train',
                7: 'truck'}


class MyObject():
    def __init__(self, frame_id, obj_id, x1, y1, w, h, obj_type='car'):
        self.init_frame_id = frame_id
        self.last_frame_id = frame_id
        self.id = obj_id
        self.x1 = x1
        self.y1 = y1
        self.w = w
        self.h = h
        self.obj_type = obj_type
        self.start_direction = 0

        self.traj = {frame_id: [x1, y1, w, h, 0, 0]}
        self.initial_pos = [x1, y1, w, h]
        self.last_pos = [x1, y1, w, h]
        self.last_mid_point = [x1+0.5*w, y1+0.5*h]

    def update(self, frame_id, obj_id, x1, y1, w, h, obj_type='car'):
        cur_mid_point = [x1+0.5*w, y1+0.5*h]
        last_move_pixel = math.sqrt((cur_mid_point[0]-self.last_mid_point[0])**2+(cur_mid_point[1]-self.last_mid_point[1])**2)
        self.traj[frame_id] = [x1, y1, w, h, last_move_pixel, frame_id-self.init_frame_id]
        if self.last_frame_id - self.init_frame_id == 20:
            vec = np.array([cur_mid_point[0] - self.initial_pos[0] - 0.5 * self.initial_pos[2],
                                cur_mid_point[1] - self.initial_pos[1] - 0.5 * self.initial_pos[3]])
            self.start_direction = vec2angle(vec)
        self.last_frame_id = frame_id
        self.last_pos = [x1, y1, w, h]
        self.last_mid_point = [x1+0.5*w, y1+0.5*h]


def angle_between(v1, v2):
    """
    Helper function which returns the unsigned angle between v1 and v2.
    The output ranges between 0 and 1, where (v1 == v2) -> 0 and (v1 == -v2) -> 1.
    """
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def vec2angle(vec):
    """
    Small helper function to translate an (x,y) vector into an angle in radians
    :param vec: np.array()
    :return: string
    """
    return round(atan2(vec[1], vec[0]), 3)


def get_statics(df, move_thresh):
    """
    Filter out static objects. For each object id we get the corresponding points, and check how far it's moved over
    it's whole lifespan. If this is less than 10px, we remove it.
    """
    ret_ids = []

    for uid in df["id"].unique():
        path = np.array(df.loc[df["id"] == uid, ["center_x", "center_y"]])
        path_segments = np.diff(path, axis=0)

        if (np.abs(path_segments.sum(axis=0)) < move_thresh).all():
            ret_ids.append(uid)

    print("Detected {} static objects".format(len(ret_ids)))

    return ret_ids



def filter_object(obj_dict):
    use_object = {}
    for key, one_obj in obj_dict.items():
        print(key, one_obj.last_frame_id)
        if one_obj.last_frame_id - one_obj.init_frame_id >= 30*3:
            use_object[key] = one_obj
    return use_object


def obj_to_df_per_img(frame_id, one_obj):
    res = one_obj.traj[frame_id]
    df = pd.DataFrame(res).T
    df.columns = ['x1', 'y1', 'w', 'h', 'last_move_pixel', 'exist_time']
    df['frame_id'] = frame_id
    df['id'] = one_obj.id
    df['cls'] = one_obj.obj_type
    df['str_angle'] = one_obj.start_direction
    df = df[['frame_id', 'id', 'x1', 'y1', 'w', 'h', 'cls', 'last_move_pixel', 'exist_time', 'str_angle']]
    return df

def output_cal_per_img(obj_dict, passing_time_dict, arr):
    column_names = ['frame_id', 'id','type', 'x1', 'y1', 'w', 'h', 'e0', 'e1', 'e2', 'e3']
    df_gb_frame = pd.DataFrame(arr, columns=column_names)
    cur_frame_id = arr[0][0]
    for cur_id in df_gb_frame['id']:
            if cur_id not in obj_dict.keys():
                se = df_gb_frame[df_gb_frame['id'] == cur_id]
                index = se.index.values[0]
                obj_dict[cur_id] = MyObject(cur_frame_id, cur_id, se.at[index,'x1'], se.at[index,'y1'], se.at[index,'w'], se.at[index,'h'], COCO_NAMES[se.at[index,'type']])
            else:
                se = df_gb_frame[df_gb_frame['id'] == cur_id]
                index = se.index.values[0]
                obj_dict[cur_id].update(cur_frame_id, cur_id, se.at[index, 'x1'], se.at[index, 'y1'], se.at[index, 'w'], se.at[index, 'h'], COCO_NAMES[se.at[index,'type']])

    for key in obj_dict.keys():
        cur_traj = obj_dict[key].traj
        init_time = min(cur_traj.keys())
        last_time = max(cur_traj.keys())
        passing_time_dict[key] = last_time - init_time

    # use_object = filter_object(obj_dict)

    list_df = []
    for one_obj in obj_dict.values():
        if cur_frame_id not in one_obj.traj.keys():
            continue
        df_tmp = obj_to_df_per_img(cur_frame_id, one_obj)
        list_df += [df_tmp]

    df_output = pd.concat(list_df).reset_index(drop=True)
    df_output = df_output.sort_values(by=['id']).reset_index(drop=True)
    return obj_dict, passing_time_dict, df_output
