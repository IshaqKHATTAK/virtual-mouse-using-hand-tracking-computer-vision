import cv2
import  numpy as np

lower = np.array([15, 150, 20])
upper = np.array([35, 255, 255])
img = cv2.imread('img2.jpg')
img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
masked = cv2.inRange(img_hsv, lower, upper)
cv2.imshow('img', img)
cv2.imshow('img', masked)
cv2.waitKey(0)