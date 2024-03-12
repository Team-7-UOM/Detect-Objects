import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch  
import numpy as np
import sys   # import libraries

sys.path.append('/home/sfr2023/yolov5') # Adds the YOLOv5 directory to the system path, making its modules accessible.

# Imports necessary components from the YOLOv5 framework:
from models.common import DetectMultiBackend # Import for YOLO model backend handling.
from utils.augmentations import letterbox # Import for image preprocessing to fit the model's input requirements.
from utils.general import non_max_suppression # Import for filtering out overlapping bounding boxes.
from utils.torch_utils import select_device # Import for device selection CPU for running the model


def scale_coords(img1_shape, coords, img0_shape):
    # Calculates scale factor based on the ratio of new shape to original shape.
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  
    # Adjusts x and y coordinates based on scale and centering.
    coords[:, [0, 2]] -= (img1_shape[1] - img0_shape[1] * gain) / 2
    coords[:, [1, 3]] -= (img1_shape[0] - img0_shape[0] * gain) / 2
    coords[:, :4] /= gain
    # Ensures coordinates are within image boundaries
    coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, img0_shape[1])
    coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, img0_shape[0])
    return coords

def detect_colors(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # Converts BGR image to HSV color space.
    # Defines ranges for different colors in HSV.
    color_ranges = {
        'red': ((0, 50, 50), (10, 255, 255)),
        'pink': ((160, 50, 50), (180, 255, 255)),
        'yellow': ((25, 50, 50), (35, 255, 255)),
        'blue': ((100, 150, 150), (140, 255, 255)),
        'purple': ((140, 50, 50), (160, 255, 255)),
        'orange': ((10, 100, 100), (25, 255, 255)),
        'green': ((35, 50, 50), (85, 255, 255)),
    }
    
    detected_colors = {}
    # Loops through the color ranges to detect colors in the image
    for color_name, (lower, upper) in color_ranges.items(): 
        mask = cv2.inRange(hsv_image, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))# Creates a mask for the current color.
        if cv2.countNonZero(mask) > 0: # Checks if the color is present in the image.
            detected_colors[color_name] = mask # Stores the mask if the color is detected.
            
    return detected_colors




class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber') # Initialize the Node with the name 'image_subscriber'
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10)
        self.bridge = CvBridge() # Creates an instance of CvBridge to convert ROS Image messages to OpenCV images.
        self.device = select_device('cpu')   # Selects the device for processing
        self.model = DetectMultiBackend('/home/sfr2023/yolov5/runs/train/exp/weights/best.pt', device=self.device)  # Loads the YOLO model from the specified path
        self.stride = int(self.model.stride.max()) if hasattr(self.model.stride, 'max') else self.model.stride # Determines the model stride for processing images.
        self.names = self.model.names # Retrieves the class names detected by the model

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8") # Converts the ROS Image message to an OpenCV image.
        except CvBridgeError as e:
            self.get_logger().error(f'Failed to convert image: {e}') # Logs an error message if conversion fails.
            return

        img = letterbox(cv_image, new_shape=640, stride=self.stride)[0]  
        img = img.transpose((2, 0, 1))[::-1] # Reorders the image channels
        img = np.ascontiguousarray(img) # Ensures the array is contiguous
        img = torch.from_numpy(img).to(self.device).float() / 255.0 # Converts the image to a PyTorch tensor and normalizes it
        if img.ndimension() == 3:
            img = img.unsqueeze(0) # Adds a batch dimension if necessary.

        pred = self.model(img, augment=False, visualize=False)   # Performs object detection on the image.
        pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False) # Applies non-maximum suppression to filter the predictions.

        orb=cv2.ORB_create() # Creates an ORB keypoint detector instance

        for i, det in enumerate(pred):
            if len(det): 
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], cv_image.shape).round() # Scales the detection coordinates back to the original image size.

                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = map(int, xyxy) # Extracts the bounding box coordinates.
                    target_region = cv_image[y1:y2, x1:x2]  # Extracts the region of interest.
                    detected_colors = detect_colors(target_region)  # Detects specific colors within the target region
                    keypoints, _ = orb.detectAndCompute(target_region, None) # Detects keypoints in the target region.
                    target_region_with_keypoints = cv2.drawKeypoints(target_region, keypoints, None, color=(0, 255, 0), flags=0) # Draws keypoints on the target region.
                    cv_image[y1:y2, x1:x2] = target_region_with_keypoints # Updates the original image with keypoints.

                    for color, mask in detected_colors.items():# For each detected color, create a highlight and display it

                        highlight = cv2.bitwise_and(target_region, target_region, mask=mask)
                        cv2.imshow(f"{color.capitalize()} Highlight", highlight)
                    
                    label = f'{self.names[int(cls)]} {conf:.2f}' # Annotates the original image with the detection label and bounding box.
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        # Displays the processed image
        cv2.imshow("Camera Image with Detections", cv_image)  
        cv2.waitKey(1)

# Main function that initializes the ROS node and spins it to start processing data.
def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)

    image_subscriber.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()
