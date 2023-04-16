import cv2
import numpy as np
from pynput.mouse import Button, Controller
from PIL import ImageGrab

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
kernel = np.ones((3, 3), np.uint8)

# Initialize the background subtractor object
fgbg = cv2.createBackgroundSubtractorMOG2()

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
cap.set(cv2.CAP_PROP_FPS, 15)

while True:
    ret, frame = cap.read()
    # Apply the background subtractor to the frame to remove the face
    fgmask = fgbg.apply(frame)
    # Apply a Gaussian blur to the background subtracted frame
    blur = cv2.GaussianBlur(fgmask, (15, 15), 0)
    eroded = cv2.erode(blur, kernel, iterations=5)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask that detects red pixels
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    # combining the two mask
    mask_red = cv2.bitwise_or(mask1, mask2)

    yellow_mask = cv2.inRange(frame, yellow_lower, yellow_upper)

    # Apply the mask to the original frame to extract the red objects
    result = cv2.bitwise_and(frame, frame, mask=mask_red)
    # Bitwise-AND mask and original image to extract yellow color
    res = cv2.bitwise_and(frame, frame, mask=yellow_mask)
    # res = cv2.erode(res, kernel=kernel, iterations=1)
    res = cv2.dilate(res, kernel=kernel, iterations=3)

    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    yellow_max = max(contours_yellow, key=cv2.contourArea)
    x_yellow, y_yellow, w_yellow, h_yellow = cv2.boundingRect(yellow_max)
    cv2.rectangle(frame, (x_yellow, y_yellow), (x_yellow + w_yellow, y_yellow+h_yellow), (0, 255, 0), 2)
    # Draw a bounding box around each detected object
    if len(contours) > 0:
        # Get the largest contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # center points of the detected regions

        print(f"camera coordinate x == {x} y== {y}")
        scaling_factor = (wscrn / wCam, hscrn / hCam)
        screen_x, screen_y = (int(x * scaling_factor[0]), int(y * scaling_factor[1]))

        # smoothing
        clocX = plocX + (screen_x - plocX) / smoothening
        clocY = plocY + (screen_y - plocY) / smoothening
        mouse = Controller()
        # mouse.move(int(wscrn - clocX), int(clocY))
        # (wscrn-screen_x) ==> flip the coordinate left right --> mirror effect
        # mouse.position = (wscrn-clocX, clocY)
        plocX, plocY = clocX, clocY
        # mouse.move(screen_x, screen_y)
        print(f'screen coordinate x_cordinate == {screen_x} and y_cordinate == {screen_y}')
        print(f'smoothened value {int(wscrn - clocX)} y_smoth = {int(clocY)}')

    cv2.imshow('Original', frame)
    cv2.imshow('Red Objects', result)
    cv2.imshow('green color', res)

    # Wait for a key press and exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()