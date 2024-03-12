import pyrealsense2 as rs
import numpy as np
import cv2
import torch #import libraires

model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/sfr2023/yolov5/runs/train/exp/weights/best.pt')# Loads the YOLOv5 model pre-trained on custom data

# Initializes the RealSense pipeline.
pipeline = rs.pipeline() 
config = rs.config()
# Configures the pipeline to stream depth and color data.
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Gets the depth sensor's scale (meters per depth unit)
depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

try:
    while True:
        frames = pipeline.wait_for_frames() # Waits for the next set of frames from the camera
        depth_frame = frames.get_depth_frame() # Gets the depth frame.
        color_frame = frames.get_color_frame() # Gets the color frame.
        if not depth_frame or not color_frame:  # Checks if either frame is unavailable.
            continue
        
        # Converts depth and color frames to numpy arrays.
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Uses the YOLOv5 model to detect objects in the color image
        results = model(color_image)
        detections = results.pandas().xyxy[0] # Converts detection results to a pandas DataFrame

        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics # Retrieves the depth frame's intrinsics (camera parameters)

        for index, row in detections.iterrows():
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']) # Extracts bounding box coordinates.
            x_center, y_center = int((xmin + xmax) / 2), int((ymin + ymax) / 2) # Calculates the center of the bounding box
            depth = depth_image[y_center, x_center] * depth_scale # Retrieves the depth at the center of the detected object.

            
            x, y, z = rs.rs2_deproject_pixel_to_point(depth_intrin, [x_center, y_center], depth) # Converts depth from pixel to 3D point coordinates.
            print(f"Object {index}: Depth: {depth:.2f}m, 3D coordinates: x={x:.3f}, y={y:.3f}, z={z:.3f} meters") 

            
            cv2.rectangle(color_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2) # Draws a rectangle around the detected object in the color image.
            depth_text = f"Depth: {depth:.2f}m" # Adds depth information text on the image.
            cv2.putText(color_image, depth_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # Displays the color image with detection bounding boxes and depth information.
        cv2.imshow('RealSense', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop() # Stops the pipeline after breaking out of the loop.

