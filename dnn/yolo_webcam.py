"""
Image detection with webcam using YOLO
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


o = yolo.YOLO(**config)
import cv2
cap = cv2.VideoCapture(0)
camera_scale = 1.

while True:
  ret, image = cap.read()
  if cv2.waitKey(10) == 27:
    break
  h, w = image.shape[:2]
  rh = int(h * camera_scale)
  rw = int(w * camera_scale)
  image = cv2.resize(image, (rw, rh))
  image = image[:,:,(2,1,0)]
  image = Image.fromarray(image)
  r_image = o.detect_image(image)
  out_img = np.array(r_image)[:,:,(2,1,0)]
  cv2.imshow("YOLOv2", np.array(out_img))
  print('\n\npress Esc key to exit, any other key to continue\n\n')
  if cv2.waitKey(0) == 27:
    break

o.close_session()

# eof
