import torch 
from torch import nn 
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from cnn import AvgPool_Modified_VGG
from cnn import eval

print("this is the test program\n")
path = input("Where is the test data?")
net = AvgPool_Modified_VGG()
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.8876], std=[0.3158]) # Data computed from the train set
])

test_data = ImageFolder(path, transform=transform)
test_loader = DataLoader(test_data, batch_size=256, shuffle=True)

ckpt_path = "./best-99.40.pth"
net.load_state_dict(torch.load(ckpt_path))
acc = eval(net,test_loader)
print(acc)