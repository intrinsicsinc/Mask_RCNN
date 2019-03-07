#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import sys
import json
import math
import random
import argparse
import skimage.io
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
from samples.coco import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    DETECTION_MIN_CONFIDENCE = 0.55

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str)
    args = parser.parse_args()
    return args

def process_batch(frames):
    """
    Segment a list of images, produce and save the output frame
    and return the next frame number with a list of person counts
    """
    global i, person_counts
    fps = 15
    results = model.detect(frames, verbose=1)

    for frame, r in zip(frames, results):
        output_frame, count = visualize.draw_frame(frame, r['rois'],
                    r['masks'], r['class_ids'], class_names, r['scores'])
        person_counts.append(count)

        fig, axs = plt.subplots(2, figsize=(10, 10), gridspec_kw = {'height_ratios':[2, 1.2]})

        height, width = frame.shape[:2]
        axs[0].imshow(output_frame, extent=(0,width,0,height), shape=(width,height))

        averages = calculate_averages(person_counts, 3*fps)
        axs[1].plot(averages)
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('# of Persons')
        axs[1].set_xticks([])
        axs[1].set_ylim(bottom=0, top=60)
        axs[1].grid(b=True)

        i += 1
        output_path = os.path.join('output', '{}.jpg'.format(i))
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

def calculate_averages(counts, n):
    length = 150
    s = pd.Series(counts[-(length+n):])
    averages = s.rolling(window=n, min_periods=1).mean().values
    return averages[-length:]

def tag_video(video):
    global i, person_counts
    # Open video with OpenCV
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise IOError('Unable to open video: {}'.format(video))

    # Open VideoWriter with OpenCV for creating output video
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # For each frame of the video
    frames = []
    while capture.isOpened():
        # Read next frame, process when batch is full
        ret, rawframe = capture.read()
        if not ret:
            break
        im = cv2.cvtColor(rawframe, cv2.COLOR_BGR2RGB)
        frames.append(im)

        if len(frames) == 2:
            process_batch(frames)
            frames = []

    capture.release()

# Parse CLI arguments
args = parse_args()

# Tag the target video
i = 0
person_counts = []
tag_video(args.video)

