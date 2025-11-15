import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_corners(np_img):
    corners = cv2.goodFeaturesToTrack(np_img,
                                      maxCorners=7,
                                      qualityLevel=0.1,
                                      minDistance=2)

    corners = np.uint64(corners)
    return corners

def create_img_np_array(p):
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


def shape_to_black_background_to_white(gray):

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