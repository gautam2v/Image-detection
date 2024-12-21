import cv2
import numpy as np
import tkinter as tk

# Initialize the video stream from the laptop's built-in webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Create a simple Tkinter window for controlling image processing parameters
root = tk.Tk()
root.title("Real-Time Dot Detection Controls")

# Create sliders for Hue, Saturation, Value ranges
hl = tk.Scale(root, from_=0, to=179, orient="horizontal", label="Hue Low", sliderlength=20)
hl.set(0)
hl.pack()

hh = tk.Scale(root, from_=0, to=179, orient="horizontal", label="Hue High", sliderlength=20)
hh.set(179)
hh.pack()

sl = tk.Scale(root, from_=0, to=255, orient="horizontal", label="Saturation Low", sliderlength=20)
sl.set(0)
sl.pack()

sh = tk.Scale(root, from_=0, to=255, orient="horizontal", label="Saturation High", sliderlength=20)
sh.set(255)
sh.pack()

vl = tk.Scale(root, from_=0, to=255, orient="horizontal", label="Value Low", sliderlength=20)
vl.set(0)
vl.pack()

vh = tk.Scale(root, from_=0, to=255, orient="horizontal", label="Value High", sliderlength=20)
vh.set(255)
vh.pack()

# Create sliders for Brightness and Contrast
brightness = tk.Scale(root, from_=-100, to=100, orient="horizontal", label="Brightness", sliderlength=20)
brightness.set(0)
brightness.pack()

contrast = tk.Scale(root, from_=0, to=200, orient="horizontal", label="Contrast", sliderlength=20)
contrast.set(100)  # 100 means no change in contrast
contrast.pack()

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

# Function to process the camera frame
def process_frame():
    ret, frame = camera.read()
    if not ret:
        print("Failed to capture frame from webcam. Exiting...")
        root.quit()
        return

    # Get slider values
    hl_value = hl.get()
    hh_value = hh.get()
    sl_value = sl.get()
    sh_value = sh.get()
    vl_value = vl.get()
    vh_value = vh.get()
    brightness_value = brightness.get()
    contrast_value = contrast.get() / 100.0  # Normalize contrast (0-200 -> 0-2)

    # Apply brightness and contrast adjustments
    adjusted_frame = cv2.convertScaleAbs(frame, alpha=contrast_value, beta=brightness_value)

    # Convert to HSV color space
    hsv = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for color filtering
    lower_bound = np.array([hl_value, sl_value, vl_value])
    upper_bound = np.array([hh_value, sh_value, vh_value])

    # Apply the mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    filtered_frame = cv2.bitwise_and(adjusted_frame, adjusted_frame, mask=mask)

    # Detect and count dots
    processed_frame = detect_dots(filtered_frame)

    # Show the processed frame
    cv2.imshow("Real-Time Dot Detection", processed_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        root.quit()

    # Keep processing frames
    root.after(10, process_frame)

# Start processing frames from the webcam
process_frame()

# Start the Tkinter event loop
root.mainloop()

# Release the camera and close all OpenCV windows when the program ends
camera.release()
cv2.destroyAllWindows()
