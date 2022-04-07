# TrafficLight-Classification-ROS-Yolov5
TrafficLight-Classification-ROS-Yolov5

# How to Use

# Object Detection
## YoloV5  
[github repo]
[weights]

# Traffic-Light-Classifier

It was considered to use state of art models as a classifier. Like EfficientNet, ViT, ConvNeXt. But due to the bug with torchScript in c++, it was worked on an old network architecture like ResNet18. A pretrained ResNet architecture was considered sufficient for performance because the problem was not complex. In this section, training scripts, metric and performance tests implemented with python. For more detailed information, you can visit [Classifier](train_module/ReadMe.md)


# Rviz Visualization
for input video please download drive link.
[test_video] (https://drive.google.com/file/d/1L5DMnn-yT53Z-xZbbVWHSwH7B8AAZTc-/view?usp=sharing)

Result or video
![ezgif com-gif-maker](https://user-images.githubusercontent.com/37477289/162274767-fbe384fe-4202-4fd9-9c7d-e63c1196c528.gif)
