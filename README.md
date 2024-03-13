# Detect_Object
These codes can be used to detect wood blocks of different shapes, colors and output 3D coordinates.
# How to use

   
   ## 1. Open the depth camera node
   ```
   ros2 launch realsense2_camera rs_launch.py depth_module.profile:=1280x720x30 pointcloud.enable:=true
   ```
   ## 2. Open Image_convert_node.py
   open the
     ```/object_detection/object_detection/Image_convert_node.py
     ``` 
     and run it in Code or Pycharm.When the depth camera captures the target image, a rectangular box marked with the class will appear in the target image, and a highlighted area will appear.
     
      
    

   
   
   
   

