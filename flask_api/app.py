import os
import urllib
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from segment_anything import SamPredictor, sam_model_registry
CHECKPOINT_PATH = os.path.join("checkpoint")

# # large model
# CHECKPOINT_NAME = "sam_vit_h_4b8939.pth"
# CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# small model
CHECKPOINT_NAME = "sam_vit_b_01ec64.pth"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"

# graphic card check
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

if not os.path.exists(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
checkpoint = os.path.join(CHECKPOINT_PATH, CHECKPOINT_NAME)
if not os.path.exists(checkpoint):
    urllib.request.urlretrieve(CHECKPOINT_URL, checkpoint)
sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint).to(DEVICE)


predictor = SamPredictor(sam)
img_path = './test_images/3.jpg'
point_w = 2600
point_h = 1200

# # 이미지 출력방법 1
# img = Image.open(img_path)
# img

# # 이미지 출력방법 2
# from google.colab.patches import cv2_imshow
# img = cv2.imread(img_path)
# cv2_imshow(img)

# 이미지 전처리
img = Image.open(img_path)
numpydata = np.asarray(img)

points_coords = np.array([[point_w, point_h], [0, 0]])
points_label = np.array([1, -1])

# 예측
predictor.set_image(numpydata)
masks, scores, _ = predictor.predict(points_coords, points_label)


# 출력
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 15))
plt.subplot(2, 3, 1) # subplot
plt.scatter(point_w, point_h, c = 'red')
plt.imshow(img)

for i in range(3):
  mask = Image.fromarray(masks[i])
  plt.subplot(2, 3, 4+i)
  plt.title(f"score: { scores[i] }")
  plt.scatter(point_w, point_h, c = 'red')
  plt.imshow(img)
  plt.imshow(mask, alpha = 0.5)
plt.show()