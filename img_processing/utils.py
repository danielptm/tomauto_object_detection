import math

import cv2
import numpy as np
from matplotlib import pyplot as plt


class Line:
    def __init__(self, p1, p2, priority):
        self.p1 = p1
        self.p2 = p2
        self.priority = priority

class Pq:
    def __init__(self):
        self.items = []



    def insert(self, item: Line):
        self.items.append(item)
        self.items.sort(key=lambda x: x.priority)

    def pop(self):
        return self.items.pop(0)

    def is_empty(self):
        return len(self.items) == 0



def create_gray_image_from_path(p):
    img = cv2.imread(p)       # Reads as BGR by default
    img_array = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    return img_array

def render_dots(img, items):
    for i in items:
        img[i[0][1], i[0][0]] = 125

def show_image(img):
    plt.imshow(img, interpolation='nearest', cmap='gray')
    plt.show()


def remove_background_and_make_foreground_white(gray):

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold to get a clean mask
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No foreground shape found")

    # Select the largest contour as the shape
    largest = max(contours, key=cv2.contourArea)

    # Create a white background
    output = np.ones_like(gray) * 255

    # Draw the shape filled in black
    cv2.drawContours(output, [largest], -1, 0, thickness=cv2.FILLED)

    return output


class ArrowParser():
    def __init__(self, black_white_img):
        self.black_white_img = black_white_img

    def get_corners(self):
        corners = cv2.goodFeaturesToTrack(self.black_white_img,
                                          maxCorners=7,
                                          qualityLevel=0.1,
                                          minDistance=2)

        corners = np.uint64(corners)
        approx = corners.tolist()
        res = self.flatten_double_array(approx)
        return res

    def flatten_double_array(self, array):
        res = []
        for a in array:
            res.append(a[0])
        return res


def get_distance_between_2_pts(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def flatten_array_all(array):
    res = []
    for a in array:
        res.append(a.p1)
        res.append(a.p2)
    return res

def get_furthest_pts(array):
    pq = Pq()
    for i in range(len(array)):
        for j in range(len(array)):
            distance = get_distance_between_2_pts(array[i], array[j])
            line = Line(array[i], array[j], distance)
            pq.insert(line)
    return [pq.items[len(pq.items) - 1], pq.items[len(pq.items) - 3], pq.items[len(pq.items) - 5]]

def set_pts(array):
    hp = None
    b1 = None
    b2 = None
    for i in range(len(array)):
        for j in range(len(array)):
            if array[i] == array[j]:
                hp = array[i]
                break
    for a in array:
        if a != hp and a != b1:
            b1 = a
        elif a != hp and a != b2:
            b2 = a
    return {'hp': hp, 'b1': b1, 'b2': b2}

