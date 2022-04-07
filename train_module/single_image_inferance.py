from torchvision import transforms
import torch
import argparse
from PIL import Image
import time
import numpy as np

import torchvision.models as models
import torch.nn as nn

def main():
    class_names = ["red","green"]

    device = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str,help="Image file Path")

    args = parser.parse_args()
    img = Image.open(args.img).convert("RGB")

    transform = transforms.Compose([transforms.RandomResizedCrop(64),transforms.ToTensor()])
    img = transform(img)
    img = img.view((1,3,64,64))

    model = models.resnet18(pretrained=True).to(device)
    model.fc = nn.Linear(512, 2).to(device) #512 for ResNet
    model.eval()
    model_weight_path = "./resnet18/best_model.pth"
    model.load_state_dict(torch.load(model_weight_path))

    model.eval()
    
    for i in range(5):
        s = time.time()
        pred = model(img.to(device))
        pred = torch.nn.functional.softmax(pred)
        pred = pred.to("cpu")
        cls = torch.max(pred, dim=1)[1]
        print(time.time()-s)

    print("Prediction : {}".format(class_names[int(cls)]))
    print("Score : {}".format(float(pred[0][int(cls)])))

if __name__ == "__main__":
    main()
