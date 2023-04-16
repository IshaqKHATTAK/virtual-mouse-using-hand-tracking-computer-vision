import cv2

# Define the lower and upper boundaries of the "yellow" color in HSV color space
yellow_lower = (20, 100, 100)
yellow_upper = (30, 255, 255)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame from BGR color space to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the yellow color using the defined lower and upper boundaries
    mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    print(mask.shape)

    # Apply a bitwise AND operation to the frame and the mask to extract only the yellow regions
    yellow_regions = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the original frame and the extracted yellow regions
    cv2.imshow('Original', frame)
    cv2.imshow('Yellow Regions', yellow_regions)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
