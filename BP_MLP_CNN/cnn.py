import torch 
from torch import nn 
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Basic LeNet-5, change sigmoid to ReLU
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(16 * 5 * 5, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 12)
    
    def forward(self,input):
        t = F.relu(self.conv1(input))
        t = self.pool1(t) 
        t = F.relu(self.conv2(t))
        t = self.pool2(t)
        t = self.flatten(t)
        t = F.relu(self.linear1(t))
        t = F.relu(self.linear2(t))
        t = self.linear3(t)
        return t
    
class VGG_Style_LeNet(nn.Module):
    def __init__(self):
        super(VGG_Style_LeNet,self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(6, 6, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(6, 12, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(12 * 7 * 7, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 12)
    
    def forward(self,input):
        t = F.relu(self.conv1(input))
        t = F.relu(self.conv2(t))
        t = self.pool1(t) 
        t = F.relu(self.conv3(t))
        t = F.relu(self.conv4(t))
        t = self.pool2(t)
        t = self.flatten(t)
        t = F.relu(self.linear1(t))
        t = F.relu(self.linear2(t))
        t = self.linear3(t)
        return t

class Modified_VGG_Style_LeNet(VGG_Style_LeNet):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(6)
        self.norm2 = nn.BatchNorm2d(6)
        self.norm3 = nn.BatchNorm2d(12)
        self.norm4 = nn.BatchNorm2d(12)
        self.norm5 = nn.BatchNorm1d(120)
        self.norm6 = nn.BatchNorm1d(84)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self,input):
        t = self.conv1(input)
        t = F.relu(self.norm1(t))
        t = self.conv2(t)
        t = F.relu(self.norm2(t))
        t = self.pool1(t) 

        t = self.conv3(t)
        t = F.relu(self.norm3(t))
        t = self.conv4(t)
        t = F.relu(self.norm4(t))
        t = self.pool2(t)

        t = self.flatten(t)
        t = self.linear1(t)
        
        t = F.relu(self.norm5(t))
        t = self.dropout1(t)
        t = self.linear2(t)
        t = F.relu(self.norm6(t))
        t = self.dropout2(t)
        t = self.linear3(t)
        return t

class AvgPool_Modified_VGG(Modified_VGG_Style_LeNet):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm2d(32)
        self.norm3 = nn.BatchNorm2d(64)
        self.norm4 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.linear_last = nn.Linear(64,12)

    def forward(self,input):
        t = self.conv1(input)
        t = F.relu(self.norm1(t))
        t = self.conv2(t)
        t = F.relu(self.norm2(t))
        t = self.pool1(t) 

        t = self.conv3(t)
        t = F.relu(self.norm3(t))
        t = self.conv4(t)
        t = F.relu(self.norm4(t))
        t = self.pool2(t)

        t = self.avgpool(t) 
        t = self.flatten(t)
        t = self.linear_last(t)
        return t

def eval(model, data_loader):
    model.eval() 
    cor = 0
    tot = 0
    with torch.no_grad(): 
        for img, y in data_loader: 
            predict_y = model(img) 
            _, predicted = torch.max(predict_y.data, 1) 
            tot += y.size(0) 
            cor += (predicted == y).sum().item()

    acc = 100 * cor / tot
    return acc

