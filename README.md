# Detect_Object
These codes can be used to detect wood blocks of different shapes, colors and output 3D coordinates.
# How to use

   
   ## 1. Open the depth camera node
   ```
   ros2 launch realsense2_camera rs_launch.py depth_module.profile:=1280x720x30 pointcloud.enable:=true
   ```
   ## 2. Open Image_convert_node.py
   Open the
     ```
   /object_detection/object_detection/Image_convert_node.py
     ``` 
     and run it in Code or Pycharm.When the depth camera captures the target image, a rectangular box marked with the class will appear in the target image, and a highlighted area will appear.
     ![image](https://github.com/QinGuo-hub/Detect_Object/blob/main/testpicture/Screenshot%20from%202024-03-13%2000-50-49.png)
     ![image](https://github.com/QinGuo-hub/Detect_Object/blob/main/testpicture/Screenshot%20from%202024-03-13%2000-52-23.png)
   ## 3.Measure the distance(depth) and output the position of target relative the depth camera
   Open the
   ```
   depth_camera.py
   ```
   
   and run it in Code or Pycharm.When the target is recognized, the code outputs the 3D coordinates of the target relative to the depth camera in the terminal and displays the distance (depth) in the real-time image.
   ![image](https://github.com/QinGuo-hub/Detect_Object/blob/main/testpicture/Screenshot%20from%202024-03-13%2001-00-25.png)
   ![image]()
   
     
      
    

   
   
   
   

