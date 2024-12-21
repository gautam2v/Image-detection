import cv2
import numpy as np
import tkinter as tk

# Replace this with the URL of your IP webcam stream
# Example: "http://192.168.1.2:8080/video"
ip_webcam_url = "http://<IP>:<Port>/video"

# Initialize the video stream from the IP webcam
camera = cv2.VideoCapture(ip_webcam_url)

# Create a simple Tkinter window for controlling image processing parameters
def nothing(x):
    pass

# Create the main window
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

# Function to process the camera frame
def process_frame():
    ret, frame = camera.read()
    if not ret:
        print("Failed to capture frame from IP webcam. Exiting...")
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
    result = cv2.bitwise_and(adjusted_frame, adjusted_frame, mask=mask)

    # Detect LED circles using Hough Circle Transform
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    dp = 1.5
    minDist = max(15, min(frame.shape[:2]) // 30)
    param1 = 60
    param2 = 8
    minRadius = 1
    maxRadius = 15

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    dot_count = 0
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        dot_count = len(circles)

        for idx, (x, y, r) in enumerate(circles, start=1):
            cv2.circle(result, (x, y), r, (0, 255, 0), 2)
            cv2.circle(result, (x, y), 2, (255, 0, 0), -1)
            cv2.putText(result, str(idx), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.putText(result, f"LEDs Detected: {dot_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Show the result in a window
    cv2.imshow("Real-Time LED Detection", result)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        root.quit()

    # Keep processing frames
    root.after(10, process_frame)

# Start processing frames from the IP webcam
process_frame()

# Start the Tkinter event loop
root.mainloop()

# Release the camera and close all OpenCV windows when the program ends
camera.release()
cv2.destroyAllWindows()
