# TrafficLight-Classification-ROS-Yolov5
TrafficLight-Classification-ROS-Yolov5

# How to Use

# Object Detection
## YoloV5  
[github repo]
[weights]

# Traffic-Light-Classifier

It was considered to use state of art models as a classifier. Like EfficientNet, ViT, ConvNeXt. But due to the bug with torchScript in c++, it was worked on an old network architecture like ResNet18. A pretrained ResNet architecture was considered sufficient for performance because the problem was not complex.

## Model

Pre-trained ResNet18 network is used. The core idea of ResNet is introducing a so-called “identity shortcut connection” that skips one or more layers. Residual Block: 
In order to solve the problem of the vanishing/exploding gradient, this architecture introduced the concept called Residual Network. In this network we use a technique called skip connections . The skip connection skips training from a few layers and connects directly to the output. 
The approach behind this network is instead of layers learn the underlying mapping, we allow network fit the residual mapping. So, instead of say H(x), initial mapping, let the network fit, F(x) := H(x) – x which gives H(x) := F(x) + x. 

## Dataset
This traffic light dataset consists of 1440 number of color images in 2 categories - red, and green. As with most human-sourced data, the data is not evenly distributed among the types. There are:

    904 red traffic light images
    536 green traffic light images

Note: All images come from this MIT self-driving car course and are licensed under a 9Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).

[training results]
[weights]

# Rviz Visualization
[test_video]
[output_video]
