import cv2
import numpy as np
import toml


def fix_barrel_distortion(image: np.ndarray, config_file_path: str) -> np.ndarray:
    
    with open('configs/camera_config.toml', 'r') as f:
        config = toml.load(f)

    newcameramtx = config['camera']['camera_matrix']
    mtx = config['camera']['mtx']
    dist = config['camera']['dist']

    h,w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    mapx,mapy=cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
    dst = cv2.remap(image, mapx,mapy,cv2.INTER_LINEAR)
    return dst