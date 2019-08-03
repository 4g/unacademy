import numpy as np
import cv2
from PIL import Image

def remove_green(image):
    img = image.copy()
    h,w = img.shape[0], img.shape[1]
    background = img[0:h//8, 0:w//8]
    green_red_ratio = background[:,:,1]/background[:,:,2]
    min_ratio = np.min(green_red_ratio)
    max_ratio = np.min(green_red_ratio)
    img_ratio = img[:,:,1]/img[:,:,2]
    img_ratio[min_ratio < img_ratio] = 0
    img_ratio = to_int(img_ratio) 
    img_ratio[img_ratio > 0] = 1
    for i in range(3):
        img[:,:,i] = image[:,:,i]*img_ratio

    return img

def to_int(im):
    return np.array(im * (255/np.max(im)), dtype=np.uint8)
