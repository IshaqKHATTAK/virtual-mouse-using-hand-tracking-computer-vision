import cv2
import numpy as np
import time


def erod_dialate_hsv(frame, kernel):
    # Initialize the background subtractor object
    fgbg = cv2.createBackgroundSubtractorMOG2()
    fgmask = fgbg.apply(frame)
    # Apply a Gaussian blur to the background subtracted frame
    blur = cv2.GaussianBlur(fgmask, (15, 15), 0)
    # erod the image to remove the noise
    eroded = cv2.erode(blur, kernel, iterations=5)
    # dialate the image to get the correct bounding box by
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    # convert the image int to hsv to image
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return hsv

def create_mask(hsv, lower_limit1, upper_limit1, lower_limit2 = None, upper_limit2 = None):
    """
    :param hsv: hsv image
    :param lower_limit1:
    :param upper_limit1:
    :param lower_limit2: if exit then take it
    :param upper_limit2:  //
    :return: mask of required color
    """
    if lower_limit2 == None:
        mask1 = cv2.inRange(hsv, lower_limit1, upper_limit1)
        return mask1
    else:
        mask1 = cv2.inRange(hsv, lower_limit1, upper_limit1)
        mask2 = cv2.inRange(hsv, lower_limit2, upper_limit2)
        # combining the two mask
        mask = cv2.bitwise_or(mask1, mask2)
        return mask

def web_to_screen(x, y, wscrn, hscrn, wcam, hcam):
    """

    :param x: webcam coordinate x
    :param y: webcam coordinate y
    :param wscrn:
    :param hscrn:
    :param wcam:
    :param hcam:
    :return: screen coordinates according the webcam coordinates
    """
    scale_factor = (wscrn/wcam, hscrn/hcam)
    screen_coordinates = (int(x*scale_factor[0]), int(y*scale_factor[1]))
    return screen_coordinates

def smoothing(screen_x, screen_y, prev_x, prev_y, smothing_fac):
    clock_x = prev_x + (screen_x - prev_x) / smothing_fac
    clock_y = prev_y + (screen_y - prev_y) / smothing_fac
    return (clock_x, clock_y)
