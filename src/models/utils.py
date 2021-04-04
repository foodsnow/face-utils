from numpy import dot
from numpy.linalg import norm
import numpy as np
import torch
import cv2


def cosine_simularity(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def sharpen_image(image):
    np_image = image.permute(1, 2, 0).numpy()
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_image = cv2.filter2D(np_image, -1, kernel)
    tensor_img = torch.from_numpy(sharpened_image)
    return tensor_img.permute(2, 0, 1)
