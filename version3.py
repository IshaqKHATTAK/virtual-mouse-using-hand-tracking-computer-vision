import cv2
import numpy as np
from pynput.mouse import Button, Controller
from PIL import ImageGrab
from tracking import erod_dialate_hsv, create_mask, web_to_screen, smoothing

# img = ImageGrab.grab()
wscrn, hscrn = 1920, 1080
wCam, hCam = 640, 480
frameR = 100 # Frame Reduction
smoothening = 7
plocX, plocY = 0, 0
clocX, clocY = 0, 0


# Define the lower and upper bounds for the red color in HSV color space
lower_red = (0, 210, 50)
upper_red = (5, 255, 255)
lower_red2 = (172, 210, 50)
upper_red2 = (175, 255, 255)

yellow_lower = (20, 100, 100)
yellow_upper = (30, 255, 255)

green_lower = (45, 100, 50)
green_upper = (62, 255, 255)

kernel = np.ones((3, 3), np.uint8)
########
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
cap.set(cv2.CAP_PROP_FPS, 15)

while True:
    ret, frame = cap.read()
    # subtracting background eroding to remove noise dialate and then  convert to hsv image formate
    hsv = erod_dialate_hsv(frame, kernel)

    # creating mask of the color in hsv image
    mask_red = create_mask(hsv, lower_red, upper_red, lower_red2, upper_red2)
    mask_yellow = create_mask(hsv, yellow_lower, yellow_upper)
    mask_green = create_mask(hsv, green_lower, green_upper)
    # Bitwise-AND mask and original image to extract yellow color
    res_red = cv2.bitwise_and(frame, frame, mask=mask_yellow)
    # res = cv2.erode(res, kernel=kernel, iterations=1)
    res_yellow = cv2.bitwise_and(frame, frame, mask=mask_yellow)
    res_green = cv2.bitwise_and(frame,frame,mask=mask_green)
    # res_yellow = cv2.dilate(res_yellow, kernel=kernel, iterations=3)

    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw a bounding box around each detected object
    if len(contours) > 0:
        mouse = Controller()
        # Get the largest contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if len(contours_yellow) > 0:
            # yellow object detected
            yellow_max = max(contours_yellow, key=cv2.contourArea)
            x_yellow, y_yellow, w_yellow, h_yellow = cv2.boundingRect(yellow_max)
            if abs(x - x_yellow) > 50:
                cv2.rectangle(frame, (x_yellow, y_yellow), (x_yellow + w_yellow, y_yellow + h_yellow), (0, 20, 0), 2)
            else:
                cv2.rectangle(frame, (x_yellow, y_yellow), (x_yellow + w_yellow, y_yellow + h_yellow), (0, 0, 225), 2)
            if (abs(x - x_yellow) or abs(y - y_yellow)) < 50:
                print('right clicked')
                # mouse.press(Button.right)
                # mouse.release(Button.right)
        if len(contours_green) > 0:
            # green object detected
            green_max = max(contours_green, key=cv2.contourArea)
            x_green, y_green, w_green, h_green = cv2.boundingRect(green_max)
            if abs(x - x_green) > 50:
                cv2.rectangle(frame, (x_green, y_green), (x_green + w_green, y_green + h_green), (0, 255, 255), 2)
            else:
                cv2.rectangle(frame, (x_green, y_green), (x_green + w_green, y_green + h_green), (0, 0, 225), 2)

            if (abs(x - x_green) or abs(y - y_green)) < 50:
                print('left clicked')
                # mouse.press(Button.left)
                # mouse.release(Button.left)

        # center points of the detected regions
        screen_x, screen_y = web_to_screen(x, y, wscrn, hscrn, wCam, hCam)
        # smoothing
        clocX, clocY = smoothing(screen_x, screen_y, plocX, plocY, smoothening)
        # (wscrn-screen_x) ==> flip the coordinate left right --> mirror effect
        # mouse.position = (wscrn-clocX, clocY)
        plocX, plocY = clocX, clocY

    cv2.imshow('Original', frame)
    cv2.imshow('Red Objects', res_red)
    cv2.imshow('yellow color', res_yellow)
    cv2.imshow('green object', res_green)

    # Wait for a key press and exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()