from image_processor import random_image, split_image_into_rows, split_row_image, find_pv_module, place_image
import matplotlib.pyplot as plt
from camera_utils import fix_barrel_distortion
from tensorflow.keras.models import load_model
import time
import cv2
import numpy as np
from typing import cast, Optional
import os
import tensorflow as tf
import random
os.environ["SM_FRAMEWORK"] = "tf.keras"


def mask_jig_saw(model_path: str, folder_paths: list, toml_path: str, show: bool = False):
    model = load_model(model_path, compile=False)
    model.compile()

    image = random_image(folder_paths)
    if image is None:
        return
    image = fix_barrel_distortion(image, toml_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cropped_image = find_pv_module(image)

    cropped_cp = np.copy(cropped_image)
    cropped_cp = cv2.pyrDown(cropped_cp)
    og_height, og_width = np.shape(cropped_cp)
    height, width = np.shape(cropped_cp)
    horz_splits = int(np.ceil(width/512)+1)
    vert_splits = int(np.ceil(height/512)+1)

    row_images = split_image_into_rows(cropped_cp, vert_splits, (512, 512))

    images = []
    for row in row_images:
        images.append(split_row_image(row, horz_splits+1, (512, 512)))

    for row in images:
        for image in row:
            temp = image["image"]
            temp = cv2.merge((temp, temp, temp))
            temp = np.expand_dims(temp, 0)
            prediction = (model.predict(temp))
            predicted_img = np.argmax(prediction, axis=3)[0, :, :]*255
            predicted_img = predicted_img.astype(np.uint8)
            predicted_img = cv2.erode(predicted_img, (20, 20))
            image["image"] = predicted_img

    masks = []
    for row in images:
        for image in row:
            temp = image["image"]
            x, y = image["x1"], image["y1"]
            blank = np.zeros((og_height, og_width))
            new = place_image(blank, temp, (y, x)).astype(np.uint8)
            masks.append(new)

    final_mask = np.zeros((og_height, og_width)).astype(np.uint8)
    print(np.shape(final_mask))
    print((og_height, og_width))
    out = cv2.VideoWriter('project.mp4', cv2.VideoWriter_fourcc(
        *'mp4v'), 3, (og_width, og_height))
    for mask in masks:
        final_mask = cv2.bitwise_or(final_mask, mask)
        # temp = final_mask
        temp = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
        # plt.imshow(temp)
        # plt.show()
        print(np.shape(temp))
        out.write(temp)
    out.release()


def rotate_image(image: np.ndarray, angle: float, scale: float):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, M, (w, h))


if __name__ == '__main__':
    model_path = "C:/Users/chuck/OneDrive/Desktop/Honors/models/resnet_backbone_512.hdf5"
    folders = ["C:/Users/chuck/OneDrive/Desktop/Honors/M0060/M0060",
               "C:/Users/chuck/OneDrive/Desktop/Honors/BT1/BT1"]
    toml_path = "C:/Users/chuck/OneDrive/Desktop/Honors/solarEL/solarel/configs/camera_config.toml"

    # image = random_image(folders)
    # image = fix_barrel_distortion(image, toml_path)
    # rot = rotate_image(image, 10, 0.95)
    # print(rot.shape)
    # plt.imshow(rot)
    # plt.show()
    # cv2.imwrite("rotated.jpg", rot)
    # cv2.imshow("test", rot)
    # cv2.waitKey(0)

    # mask_jig_saw(model_path=model_path, folder_paths=folders,
    #              toml_path=toml_path, show=True)
