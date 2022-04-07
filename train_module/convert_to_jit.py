import torch
import torchvision 
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn

#model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 2)
model.eval()
model_weight_path = "./resnet18/best_model.pth"
model.load_state_dict(torch.load(model_weight_path))

script_model = torch.jit.script(model)
script_model.save("resnet18_jit.pt")
