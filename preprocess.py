import warnings
import sys
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from skimage.draw import circle, line, line_aa, polygon
import pandas as pd
from tqdm import tqdm
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse

import torch

import face_alignment

import p2p
from p2p.config import args
from p2p.utils import AverageMeter, createOptim
from p2p.video_reader import VideoReader

device = 'cuda:0'
out_dir = '{}/{}'.format(args['root_dir'], args['data_dir'])
video_path = args['video_path']
csv_path = '{}/{}'.format(args['root_dir'], args['meta_data'])

warnings.filterwarnings("ignore")


COLORS = [[255, 0, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [0, 255, 255],
          [0, 170, 255], [0, 0, 255], [170, 0, 255], [255, 0, 255]]

pose_joints = {'face': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
               'l_brow': [17, 18, 19, 20, 21],
               'r_brow': [22, 23, 24, 25, 26],
               'bridge': [27, 28, 29, 30],
               'nostrils': [31, 32, 33, 34, 35],
               'l_eye': [36, 37, 38, 39, 40, 41, 36],
               'r_eye': [42, 43, 44, 45, 46, 47, 42],
               'outer_mouth': [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 48],
               'inner_mouth': [60, 61, 62, 63, 64, 65, 66, 67, 60]}

fan = face_alignment.FaceAlignment(
    face_alignment.LandmarksType._3D, flip_input=False, device=device)
vr = VideoReader()


def draw_pose_from_cords(face_preds, img_size, radius=2, draw_joints=True):
    colors = np.zeros(shape=img_size + (3, ), dtype=np.uint8)
    mask = np.zeros(shape=img_size, dtype=bool)
    preds = np.asarray(face_preds).reshape(68, 3).astype(np.uint8)
    for idx, v in enumerate(pose_joints.values()):
        #
        for i_pt in range(len(v)-1):
            yy, xx = line(preds[v[i_pt], 1], preds[v[i_pt], 0],
                          preds[v[i_pt+1], 1], preds[v[i_pt+1], 0])
            colors[yy, xx] = COLORS[idx]
            mask[yy, xx] = True
            yy, xx = circle(
                preds[v[i_pt], 1], preds[v[i_pt], 0], radius=radius, shape=img_size)
            colors[yy, xx] = COLORS[idx]
            mask[yy, xx] = True

        yy, xx = circle(preds[v[-1], 1], preds[v[-1], 0],
                        radius=radius, shape=img_size)
        colors[yy, xx] = COLORS[idx]
        mask[yy, xx] = True

    return colors, mask


def save_frames(frames, save_path):
    for idx in range(frames.shape[0]):
        im = Image.fromarray(frames[idx])
        im.save('{}/frame_{}.jpg'.format(save_path, idx))


def save_poses(lmk, save_path):
    for idx in range(len(lmk)):
        colors, mask = draw_pose_from_cords(lmk[idx], (224, 224))
        im = Image.fromarray(colors)
        im.save('{}/pose_{}.jpg'.format(save_path, idx))


def predict_on_video_set(df, num_workers=5):

    def process_video(i):
        entry = df.iloc[i]
        save_path = '{}/{}/{}'.format(out_dir, entry['name'], entry['snippet'])
        frames, frame_id = vr.read_frames(
            '{}/{}/{}'.format(video_path, entry['name'], entry['snippet']), 5)
        frames_t = torch.tensor(frames).permute(0, 3, 1, 2)
        try:
            lmk = fan.get_landmarks_from_batch(frames_t.cuda())
        except:
            pass
        Path(save_path).mkdir(parents=True, exist_ok=True)
        save_frames(frames, save_path)
        save_poses(lmk, save_path)

    with ThreadPoolExecutor(max_workers=5) as ex:
        tqdm(ex.map(process_video, range(len(df))), total=len(df))


if __name__ == '__main__':

    df = pd.read_csv(csv_path)
    predict_on_video_set(df)
