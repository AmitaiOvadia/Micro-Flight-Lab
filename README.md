<h1>
  Lab Code
</h1>

- **This is an example of a raw movie that comes out of the lab cameras system in [Micro-Flight Lab](https://www.beatus-lab.org/)**
- **My goal is to turn these vidoes into a simple 3D simulation of the flies' flight**
- I do it using **Python** for the deep learning pipeline in 2D, and **MATLAB** for the 3D analysis part. 
  

https://github.com/AmitaiOvadia/Micro-Flight-Lab/assets/101287679/371827fa-6307-4b1d-a0cb-cc824d7cfc20


-  **This is done by detecting interest points on the wings and body in each camera, using a deep learning pipleline**
-  First I detect the wings using a YOLOV8 instance segmentation model trained on a custom dataset
-  Then, I feed the image + wing mask into a trained pose estimation CNN to find feature points on the body and wings:
  
https://github.com/AmitaiOvadia/Micro-Flight-Lab/assets/101287679/02db3f7f-05dc-4b3e-93ae-7f5ce32bedf7

- The last step is turing this noisy 4-view detection of each point, into a 3D representation. 
- This is done by deciding each frame, based on various parameters and considerations, which 2 cameras have the best view of the point and triangulate them using the known camera matrices

https://github.com/AmitaiOvadia/Micro-Flight-Lab/assets/101287679/36312bc5-5c23-465d-b67f-c4249900ffb0

- From this 3D detections, we I can extract flight data as the wing and body pitch, roll and yaw.
