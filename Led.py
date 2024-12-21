import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk

# Specify the correct folder containing the images
folder_path = r"C:\Users\DELL\Desktop\Image detection"  # Updated folder path

# Supported image formats
image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
image_files = [f for f in os.listdir(folder_path) if f.split('.')[-1].lower() in image_extensions]

# Initialize image variable to hold the image data
image = None
current_image_index = 0  # Index for the current image

# Create a simple Tkinter window for controlling image processing parameters
def nothing(x):
    pass

# Create the main window
root = tk.Tk()
root.title("Image Detection Controls")

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

# Function to process the image with OpenCV based on the slider values
def process_image():
    global image

    # Get the slider values
    hl_value = hl.get()
    hh_value = hh.get()
    sl_value = sl.get()
    sh_value = sh.get()
    vl_value = vl.get()
    vh_value = vh.get()
    
    # Get brightness and contrast slider values
    brightness_value = brightness.get()
    contrast_value = contrast.get() / 100.0  # Normalize contrast (0-200 -> 0-2)

    # Apply brightness and contrast adjustments
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast_value, beta=brightness_value)

    # Convert to HSV color space
    hsv = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds based on slider values
    lower_bound = np.array([hl_value, sl_value, vl_value])
    upper_bound = np.array([hh_value, sh_value, vh_value])

    # Apply the mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(adjusted_image, adjusted_image, mask=mask)

    # Detect LED circles using Hough Circle Transform
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    dp = 1.5
    minDist = max(15, min(adjusted_image.shape[:2]) // 30)
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
        coordinates = []

        for idx, (x, y, r) in enumerate(circles, start=1):
            cv2.circle(result, (x, y), r, (0, 255, 0), 2)
            cv2.circle(result, (x, y), 2, (255, 0, 0), -1)
            cv2.putText(result, str(idx), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            coordinates.append((x, y))

        print(f"Detected {dot_count} LEDs.")
    else:
        print("No LEDs detected.")

    # Update the image without creating a new window
    cv2.putText(result, f"LEDs Detected: {dot_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("LED Detection", result)  # Just update the same window

# Function to load and process the image
def load_and_process_image(image_file):
    global image
    # Construct the full path to the image
    image_path = os.path.join(folder_path, image_file)

    # Load the image
    image = cv2.imread(image_path)

    # Resize the image to fit within the window if necessary
    screen_width = 800
    screen_height = 600

    height, width = image.shape[:2]
    aspect_ratio = width / height

    if width > screen_width or height > screen_height:
        if width / screen_width > height / screen_height:
            scale = screen_width / width
        else:
            scale = screen_height / height

        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))

    # Initial image processing
    process_image()

# Function to load the next image
def load_next_image():
    global current_image_index
    current_image_index = (current_image_index + 1) % len(image_files)  # Loop back to the first image when the last is reached
    load_and_process_image(image_files[current_image_index])

# Start by processing the first image in the folder
if image_files:
    load_and_process_image(image_files[0])

# Add a button to move to the next image
next_image_button = tk.Button(root, text="Next Image", command=load_next_image)
next_image_button.pack()

# Bind sliders to update image in real-time
hl.config(command=lambda x: process_image())
hh.config(command=lambda x: process_image())
sl.config(command=lambda x: process_image())
sh.config(command=lambda x: process_image())
vl.config(command=lambda x: process_image())
vh.config(command=lambda x: process_image())
brightness.config(command=lambda x: process_image())
contrast.config(command=lambda x: process_image())

# Start the Tkinter event loop
root.mainloop()
