# Traffic-Light-Classifier

You can edit the hyper parameters from the "train_config.yaml" file

You can follow the step below for training
```bash
python train.py
```
Script returns precision recall etc. params according to the results produced from test data 
```bash
python test_model.py
```
for single image inferance 
```bash
python single_image_inferance.py --img [IMG_PATH]
```

## Model

Pre-trained ResNet18 network is used. The core idea of ResNet is introducing a so-called “identity shortcut connection” that skips one or more layers. Residual Block: 
In order to solve the problem of the vanishing/exploding gradient, this architecture introduced the concept called Residual Network. In this network we use a technique called skip connections . The skip connection skips training from a few layers and connects directly to the output. 
The approach behind this network is instead of layers learn the underlying mapping, we allow network fit the residual mapping. So, instead of say H(x), initial mapping, let the network fit, F(x) := H(x) – x which gives H(x) := F(x) + x. 

## Dataset
This traffic light dataset consists of 1440 number of color images in 2 categories - red, and green. As with most human-sourced data, the data is not evenly distributed among the types. There are:

    904 red traffic light images
    536 green traffic light images

Note: All images come from this MIT self-driving car course and are licensed under a 9Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).

## Traning

![plot](https://user-images.githubusercontent.com/37477289/162275223-83e7adee-cd9a-4916-81ee-97fbe2512de8.png)

| Precision  | Precision | Recall | f1_score | test_acc | avg_inferance_time (sec) |
| ---------- | ----------|------ | --------- | -------- |------------------------- |
| ResNet18 | 0.967 | 0.972  | 0.969    | 0.961    | 0.0023 |


## Weights
[.pth](https://drive.google.com/file/d/1M-hP8YiNSJ-cW4kcN12XGqIwpb_AU-EJ/view?usp=sharing)

Torch Script version

[.pt](https://drive.google.com/file/d/1fdGC-SDkp-kBz2C_qfGGmTXh8K_RtvqE/view?usp=sharing)
