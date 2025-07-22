#!/usr/bin/env python
# coding=utf-8
"""Realtime multi-camera inference script for BEVFormer.

This script loads a BEVFormer model and performs realtime
inference on multiple camera streams. It accepts camera
indices or video file paths as input and prints the
prediction results. Optionally it can display the camera
feed and save results.
"""

import argparse
import os
import os.path as osp
import sys
from typing import List

import cv2
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate, scatter

FILE_DIR = osp.dirname(osp.abspath(__file__))
ROOT_DIR = osp.dirname(FILE_DIR)
sys.path.append(ROOT_DIR)

from projects.mmdet3d_plugin.datasets.pipelines import Compose
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Realtime inference with multi-camera input')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--cams',
        default='0',
        help='camera ids or video paths separated by comma, e.g. "0,1"')
    parser.add_argument('--device', default='cuda:0', help='device id')
    parser.add_argument('--save-dir', default=None, help='directory to save images')
    parser.add_argument('--no-show', action='store_true', help='do not display window')
    parser.add_argument('--fps', type=float, default=10.0, help='refresh rate')
    return parser.parse_args()


def build_data_pipeline(cfg) -> Compose:
    pipeline = []
    for step in cfg.data.test.pipeline:
        if step['type'] != 'LoadMultiViewImageFromFiles':
            pipeline.append(step)
    return Compose(pipeline)


def open_cameras(cam_str: str) -> List[cv2.VideoCapture]:
    cams = []
    for src in cam_str.split(','):
        src = src.strip()
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open camera source: {src}')
        cams.append(cap)
    return cams


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.to(args.device)
    model.eval()

    pipeline = build_data_pipeline(cfg)

    caps = open_cameras(args.cams)

    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                return
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        data = dict(img=frames, lidar2img=[np.eye(4) for _ in frames])
        data = pipeline(data)
        data = collate([data], samples_per_gpu=1)
        data = scatter(data, [args.device])[0]
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        print(result)

        if not args.no_show:
            for idx, frame in enumerate(frames):
                cv2.imshow(f'cam{idx}', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(int(1000 / args.fps)) & 0xFF == ord('q'):
                break
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
