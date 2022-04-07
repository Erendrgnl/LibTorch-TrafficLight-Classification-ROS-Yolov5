#@dataset https://github.com/asagar60/Udacity-ITSDC-Traffic-Light-Classifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from tqdm import tqdm
import sys
import argparse
import yaml
import matplotlib.pyplot as plt

from dataset_loader import CustomImageDataset

def train_one_epoch(model, optimizer, loss_function,data_loader, device, epoch):
    model.train()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data

        pred = model(images.to(device))

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return mean_loss.item()

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    total_num = len(data_loader.dataset)

    sum_num = torch.zeros(1).to(device)

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

    return sum_num.item() / total_num

def get_optimizer(model,train_parameters):
    name = train_parameters["optimizer"]
    lr = train_parameters["learning_rate"]
    lr_momentum = train_parameters["learning_momentum"]
    weight_decay = train_parameters["weight_decay"]

    if name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1E-4)
    if name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1E-4)
    return optimizer

def get_scheduler(optimizer,train_parameters):
    step_size = train_parameters["step_size"]
    gamma = train_parameters["learning_decay_gamma"]
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    return scheduler

def main():
    with open(r'train_config.yaml') as config:
        train_parameters = yaml.full_load(config)

    device = train_parameters["device"]
    epochs = train_parameters["epochs"]
    batch_size = train_parameters["batch_size"]

    data_transform = transforms.Compose([transforms.RandomResizedCrop(train_parameters["resize"])])
    train_data = CustomImageDataset("dataset/train",transform=data_transform)
    train_loader = DataLoader(dataset = train_data, batch_size=train_parameters["batch_size"], shuffle=True )

    val_data = CustomImageDataset("dataset/val",transform=data_transform)
    val_loader = DataLoader(dataset = val_data, batch_size = 1)

    model = models.resnet18(pretrained=True).to(device)
    model.fc = nn.Linear(512, train_parameters["num_class"]).to(device) #512 for ResNet

    # loss function
    loss_function = nn.CrossEntropyLoss()
    # optimizer
    optimizer = get_optimizer(model,train_parameters)
    # scheduler
    scheduler = get_scheduler(optimizer,train_parameters)

    best_acc = 0
    val_acc = []
    train_loss = []

    for epoch in range(epochs):      
        loss = train_one_epoch(model,optimizer,loss_function,train_loader,device,epoch)
        train_loss.append(loss)

        scheduler.step()

        acc = evaluate(model=model,
                        data_loader=val_loader,
                        device=device)
        val_acc.append(acc)
        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        if(acc > best_acc):
            best_acc = acc
            torch.save(model.state_dict(), "./resnet18/model-{}.pth".format(epoch))
            torch.save(model.state_dict(), "./resnet18/best_model.pth".format(epoch))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss)
    plt.title("train_loss")

    plt.subplot(1, 2, 2)
    plt.plot(val_acc)
    plt.title("val_acc")

    plt.savefig('resnet18.png')
    plt.show()


if __name__ == "__main__":  
    main()
