import cv2

# Define the lower and upper bounds for the red color in HSV color space
lower_red = (0, 180, 50)
upper_red = (8, 255, 255)
lower_red2 = (170, 180, 50)
upper_red2 = (178, 255, 255)

# Initialize the background subtractor object
fgbg = cv2.createBackgroundSubtractorMOG2()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # Apply the background subtractor to the frame to remove the face
    fgmask = fgbg.apply(frame)
    # Apply a Gaussian blur to the background subtracted frame
    blur = cv2.GaussianBlur(fgmask, (15, 15), 0)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask that detects red pixels
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Apply the mask to the original frame to extract the red objects
    result = cv2.bitwise_and(frame, frame, mask=mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw a bounding box around each detected object
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Original', frame)
    cv2.imshow('Red Objects', result)


    # Wait for a key press and exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
