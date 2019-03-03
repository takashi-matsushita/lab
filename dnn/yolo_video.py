"""
image detection with video using YOLO
by Takashi MATSUSHITA
"""

import numpy as np
from PIL import Image
import yolo

config = {
  "model_path": 'model_data/yolo-tiny.h5',
  "anchors_path": 'model_data/tiny_yolo_anchors.txt',
  "classes_path": 'model_data/coco_classes.txt',
  "score" : 0.3,
  "iou" : 0.45,
  "model_image_size" : (416, 416),
  "gpu_num" : 1,
  }


path = 'some_interesting_video.mp4'
output = 'test.mov'

o = yolo.YOLO(**config)

yolo.detect_video(o, path, output)
