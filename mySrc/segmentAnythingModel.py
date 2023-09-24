import os
import urllib
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from segment_anything import SamPredictor, sam_model_registry
import matplotlib.pyplot as plt

class segmentAnythingModel:

    def __init__(self):
        # # large model
        # CHECKPOINT_NAME = "sam_vit_h_4b8939.pth"
        # CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

        # small model
        CHECKPOINT_NAME = "sam_vit_b_01ec64.pth"
        CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        MODEL_TYPE = "vit_b"

        CHECKPOINT_PATH = os.path.join("checkpoint")

        # graphic card check
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(DEVICE)

        if not os.path.exists(CHECKPOINT_PATH):
            os.makedirs(CHECKPOINT_PATH, exist_ok=True)
        checkpoint = os.path.join(CHECKPOINT_PATH, CHECKPOINT_NAME)
        if not os.path.exists(checkpoint):
            urllib.request.urlretrieve(CHECKPOINT_URL, checkpoint)

        self.sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint).to(DEVICE)

    def getValidArea(self, ndarray):
        row, col = np.where(ndarray == True)
        minRow, maxRow = row.min(), row.max()
        minCol, maxCol = col.min(), col.max()
        return [minRow, maxRow, minCol, maxCol]


    def onePointSegment(self, imgPath, point_x, point_y):
        predictor = SamPredictor(self.sam)

        # 이미지 전처리
        img = cv2.imread(imgPath)

        points_coords = np.array([[point_x, point_y], [0, 0]])
        points_label = np.array([1, -1])

        # 예측
        predictor.set_image(img)

        masks, scores, _ = predictor.predict(points_coords, points_label)


        # 이미지 마스킹

        #max_index = np.argmax(scores)
        max_index = 0

        # 크롭을 위한 범위 측정
        minRow, maxRow, minCol, maxCol = self.getValidArea(masks[max_index])

        # datatype boolean -> unit 8, true -> 255
        binary_1ch = np.asarray(masks[max_index], dtype="uint8") * 255

        # 이미지와 bitwise 연산을 위해 rgb 3 channel로 변환
        binary_3ch = cv2.merge((binary_1ch, binary_1ch, binary_1ch))

        # 흰배경 만들기 위해 마스크 역변환 후 or 연산
        inverted_mask = cv2.bitwise_not(binary_3ch)
        masked_array = cv2.bitwise_or(img, inverted_mask)

        # 크롭
        cropped_array = masked_array[minRow:maxRow, minCol:maxCol, 0:4]

        # np array -> image
        masked_img = Image.fromarray(cropped_array)

        return masked_img

# point_x, point_y = input().split(" ")
# point_x = int(point_x)
# point_y = int(point_y)

# seg = segmentation()
# result_dict = seg.onePointSegment(point_x, point_y)

# import matplotlib.pyplot as plt
# plt.imshow(result_dict[1])
# plt.show()
# # cv2.imshow("test", result_dict[1])
# # input()