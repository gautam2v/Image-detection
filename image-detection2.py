import cv2
import numpy as np

# Function to detect and count dots
def detect_dots(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Use Hough Circle Transform to detect circles (dots)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,  # Inverse ratio of resolution
        minDist=20,  # Minimum distance between circles
        param1=50,  # Upper threshold for edge detection
        param2=30,  # Threshold for center detection
        minRadius=5,  # Minimum radius of circles
        maxRadius=30  # Maximum radius of circles
    )

    dot_count = 0
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        dot_count = len(circles)

        # Draw detected circles on the frame
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)  # Outer circle
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)  # Center point

    # Display the number of dots on the frame
    cv2.putText(frame, f"Dots Count: {dot_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame

# Access the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Process the frame to detect and count dots
    processed_frame = detect_dots(frame)

    # Show the processed frame
    cv2.imshow("Dot Detection", processed_frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()