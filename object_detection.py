import pyrealsense2 as rs
import numpy as np
import cv2
import torch #import libraires

def color_name(hsv_img):
   # Define a dictionary of color ranges in the HSV color space.
    colors = {
        'red': ((0, 50, 50), (10, 255, 255)),
        'pink': ((170, 50, 50), (180, 255, 255)),
        'yellow': ((25, 50, 50), (35, 255, 255)),
        'blue': ((100, 50, 50), (130, 255, 255)),
        'purple': ((140, 50, 50), (160, 255, 255)),
        'green': ((50, 50, 50), (70, 255, 255))
    }
    
    cname = "unknown" # Initialize color name as "unknown".

    
    for color, (lower, upper) in colors.items():
        lower = np.array(lower, np.uint8) # Convert lower range to numpy array.
        upper = np.array(upper, np.uint8) # Convert upper range to numpy array
        mask = cv2.inRange(hsv_img, lower, upper) # Create a mask for pixels within the color range.
        if mask.any():  # Check if any pixel matches the color range
            cname = color # Assign the color name if a match is found.
            break 

    return cname

# Load a trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/sfr2023/yolov5/runs/train/exp/weights/best.pt')

# Initialize the RealSense camera pipeline
pipeline = rs.pipeline()
config = rs.config()
# Configure the pipeline to stream color images.
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames() # Wait for the next frame.
        color_frame = frames.get_color_frame() # Get the color frame.
        if not color_frame: # Check if the color frame is available.
            continue

        color_image = np.asanyarray(color_frame.get_data()) # Convert the color frame to a numpy array.

        # Detect objects in the color image using the loaded model.
        results = model(color_image)
        results.render()

  
        for det in results.xyxy[0]:
            xmin, ymin, xmax, ymax, conf, cls = int(det[0]), int(det[1]), int(det[2]), int(det[3]), det[4], det[5] # Extract bounding box coordinates, confidence score, and class id.
            obj_img = color_image[ymin:ymax, xmin:xmax] # Crop the detected object from the color image.
            hsv_obj_img = cv2.cvtColor(obj_img, cv2.COLOR_BGR2HSV)   # Convert the object image to HSV color space.
            cname = color_name(hsv_obj_img) # Identify the dominant color of the object.
            cv2.putText(color_image, cname, (xmin, ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)# Display the identified color name on the color image.
        # Display the color image with bounding boxes and color names
        cv2.imshow('RealSense Color Stream', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()

