from dataset_loader import CustomImageDataset
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms
import torch
import torch.nn as nn

def main():
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    device = "cuda"

    data_transform = transforms.Compose([transforms.RandomResizedCrop(64)])
    test_data = CustomImageDataset("dataset/test",transform=data_transform)
    test_loader = DataLoader(dataset = test_data, batch_size = 1)

    model = models.resnet18(pretrained=True).to(device)
    model.fc = nn.Linear(512, 2).to(device) #512 for ResNet
    model.eval()
    model_weight_path = "./resnet18/best_model.pth"
    model.load_state_dict(torch.load(model_weight_path))


    model.eval()
    
    total_num = len(test_loader.dataset)
    sum_num = torch.zeros(1).to(device)

    for images, labels in test_loader:
        pred = model(images.to(device))
        pred = torch.max(pred, dim=1)[1]
        pred = pred.to("cpu")
        if(pred==0 and labels ==0):
            tp += 1
        elif(pred==1 and labels == 1):
            tn += 1
        elif(pred==0 and labels == 1):
            fp += 1
        elif(pred==1 and labels == 0):
            fn += 1   
       
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f1_score = 2* ((precision * recall) / (precision+recall))
    test_acc = (tp+tn) / total_num
    print("Precision {}".format(precision))
    print("Recall {}".format(recall))
    print("f1_score {}".format(f1_score))
    print("test_acc {}".format(test_acc))


if __name__ == "__main__":
    main()
