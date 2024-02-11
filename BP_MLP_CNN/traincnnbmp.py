import torch 
from torch import nn 
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from cnn import LeNet,VGG_Style_LeNet,Modified_VGG_Style_LeNet,AvgPool_Modified_VGG
from cnn import eval

# net = LeNet()
# net = VGG_Style_LeNet()
# net = Modified_VGG_Style_LeNet()
net = AvgPool_Modified_VGG()
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.8876], std=[0.3158]) # Data computed from the train set
])

# data = ImageFolder(r'./train', transform=transform)
# train_loader = DataLoader(data, batch_size=256, shuffle=True)
# # Note: data type must be numpy.ndarray
# # example of data shape: (50000, 32, 32, 3). Channel is last dimension

# # find mean and std for each channel, then put it in the range 0..1
# mean = np.round(data.mean(axis=(0,1,2))/255,4)
# std = np.round(data.std(axis=(0,1,2))/255,4)
# print(f"mean: {mean}\nstd: {std}")

train_data = ImageFolder(r'./train', transform=transform)
val_data = ImageFolder(r'./validation', transform=transform)


train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

max_iter = 150
cross_entropy = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=max_iter )



train_acc = []
val_acc = []
max_acc = 0  

for epoch in range(max_iter):
    for i, (img, y) in enumerate(train_loader):    
        predict_y = net(img)
        loss = cross_entropy(predict_y, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    train_acc.append(train:=eval(net, train_loader))
    val_acc.append(val:=eval(net, val_loader))

    if val > max_acc:
        max_acc = val
    torch.save(net.state_dict(), f'val{max_acc}.pth')
        
    print(f'epoch {epoch}/{max_iter} loss: {loss.item()} train acc: {train}, val acc: {val}')
print(max_acc)


plt.figure(figsize=(12, 6))
plt.plot(range(1, max_iter+1), train_acc, color='#3464f2', label='train acc')
plt.plot(range(1, max_iter+1), val_acc, color='#e474f2', label='val acc')
plt.xlabel('epoch')
plt.ylabel('Acc')
plt.title('Train Vs Validation Accuracy')
plt.legend()
plt.show()


