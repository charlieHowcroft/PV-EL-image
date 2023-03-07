import cv2
import numpy as np
import toml
import matplotlib.pyplot as plt


def fix_barrel_distortion(image: np.ndarray, config_file_path: str) -> np.ndarray:
    with open(config_file_path, 'r') as f:
        config = toml.load(f)

    newcameramtx = np.array(config['camera']['camera_matrix'], np.float32)
    mtx = np.array(config['camera']['mtx'], np.float32)
    dist = np.array(config['camera']['dist'], np.float32)

    h, w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(
        mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
    return dst
