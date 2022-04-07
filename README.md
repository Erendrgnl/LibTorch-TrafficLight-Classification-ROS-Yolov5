# TrafficLight-Classification-ROS-Yolov5
TrafficLight-Classification-ROS-Yolov5

# Object Detection ROS Package

Please change libTorch Path in Cmake to compile correctly.

    #Line 18 in CMakeLists.txt
    set(CMAKE_PREFIX_PATH "/home/eren/Documents/libtorch_cu11.1/share/cmake/Torch/")

"traffic_light_classification" is a ros package. "data_handler.cpp" read video from directory and publish as a ros node which name is /data_handler. "classifier.cpp" first detect objects if object is a traffic light call resNet18 model to classify traffic light color. Then draws bbox on detected area and publish image on /detection_result node.

## YoloV5  
With the original authors work on YOLO coming to a standstill, YOLOv4 was released by Alexey Bochoknovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao. The paper was titled YOLOv4: Optimal Speed and Accuracy of Object Detection. Shortly after the release of YOLOv4 Glenn Jocher introduced YOLOv5 using the Pytorch framework.

[yolov4 paper summary](https://towardsdatascience.com/yolo-v4-optimal-speed-accuracy-for-object-detection-79896ed47b50)

for weight and referance codes please visit link below

[YOLOv5-LibTorch](https://github.com/Nebula4869/YOLOv5-LibTorch.git)


# Traffic-Light-Classifier

It was considered to use state of art models as a classifier. Like EfficientNet, ViT, ConvNeXt. But due to the bug with torchScript in c++, it was worked on an old network architecture like ResNet18. A pretrained ResNet architecture was considered sufficient for performance because the problem was not complex. In this section, training scripts, metric and performance tests implemented with python. For more detailed information, you can visit [Classifier](train_module/ReadMe.md)


# Rviz Visualization
for input video please download drive link.

[input_video] (https://drive.google.com/file/d/1L5DMnn-yT53Z-xZbbVWHSwH7B8AAZTc-/view?usp=sharing)

![ezgif com-gif-maker](https://user-images.githubusercontent.com/37477289/162274767-fbe384fe-4202-4fd9-9c7d-e63c1196c528.gif)
