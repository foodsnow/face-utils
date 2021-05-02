import imutils
from numpy import dot
from numpy.linalg import norm
import numpy as np
import torch
import cv2
from torchvision import transforms


def cosine_simularity(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def sharpen_image(image):
    np_image = image.permute(1, 2, 0).numpy()
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_image = cv2.filter2D(np_image, -1, kernel)
    tensor_img = torch.from_numpy(sharpened_image)
    return tensor_img.permute(2, 0, 1)


def get_labels(model, digits):
    digits = transforms.Normalize(0.5, 0.5)(digits)
    model.eval()
    with torch.no_grad():
        output = model(digits)
        pred = output.max(1, keepdim=True)
        pred_indices = pred[1]
        return pred_indices.numpy()


def process_iin_image(model, image):
    iin = image / 255
    num_digits = 12
    height, width = iin.shape
    width_per_digit = width // num_digits
    digits = []
    for i in range(12):
        digit = iin[:, i * width_per_digit: (i + 1) * width_per_digit]
        digit = cv2.resize(digit, (28, 28))
        digit = digit[np.newaxis, ...]
        digits.append(digit)
    digits = np.stack(digits)
    digits = torch.from_numpy(digits).float()
    return get_labels(model, digits).tolist()


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def get_iin_from_document(image):
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    warped = (warped > thresh).astype("uint8") * 255

    h, w = warped.shape

    if h > w:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    h, w = warped.shape
    first_half = warped[:, 0:w // 2]
    second_half = warped[:, w // 2:w]
    first_half_density = np.sum(first_half)
    second_half_density = np.sum(second_half)

    if first_half_density < second_half_density:
        warped = cv2.rotate(warped, cv2.ROTATE_180)

    warped = cv2.resize(warped, (500, 320))

    h, w = warped.shape
    top = int(h - 0.1091 * h)
    bottom = int(h - 0.0203 * h)
    start = int(0.1428 * w)
    end = int(0.4418 * w)

    iin = warped[top:bottom, start:end]
    return iin
