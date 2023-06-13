import numpy as np
import cv2


def draw_contour(img, mask):
    img_copy = img.copy()
    x, y, width, height = cv2.boundingRect(mask)
    cv2.rectangle(img_copy, (x, y), (x + width, y + height), (255, 255, 0), 1)
    cv2.drawContours(img_copy, [mask], -1, (0, 0, 255), 1)
    return img_copy