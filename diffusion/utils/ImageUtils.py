import cv2
import numpy as np
from scipy import ndimage


def get_image(url):
    return cv2.imread(url)


def normalize_image(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def apply_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_noise(img):
    noise = np.random.normal(0.01, 0.03, img.shape)
    img_noise = img + noise
    return np.clip(img_noise, 0, 1)


def apply_trunc(img, f):
    img[1:-1, 1:-1] += f()
    return img


def apply_gradient(img):
    return np.sqrt(ndimage.sobel(img, axis=0) ** 2 + ndimage.sobel(img, axis=1) ** 2)
