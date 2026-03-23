from torchvision import datasets,transforms
from torch.utils.data import DataLoader,random_split
import torch
import torch.nn as nn
import torch.optim as optim

transform = transforms.Compose([
    transforms.RandomCrop(32,padding=4),#位置ズレ
    transforms.RandomHorizontalFlip(),#反転(data augmentation)
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(),#uint8 → float32 [0,255] → [0,1]
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))#-1~1
])

train_data = datasets.CIFAR10(
    root = 'data_cifar10',
    download = True,
    train = True,
    transform = transform
)
test_data = datasets.CIFAR10(
    root = 'data_cifar10',
    download = True,
    train = False,
    transform = transform
)
train_size = int(0.8*len(train_data))
val_size = len(train_data)-train_size

train_data,val_data = random_split(train_data,[train_size,val_size])

trainloader = DataLoader(train_data,batch_size=64,shuffle=True)
valloader = DataLoader(val_data,batch_size=64)
testloader = DataLoader(test_data,batch_size=64)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3,padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2,2)

        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Linear(64*8*8,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)
    
    def forward(self,x):
        x = self.pool(torch.relu(self.bn1((self.conv1(x)))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x,1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

def evaluate(model,loader):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for X,y in loader:
            X,y = X.to(device),y.to(device)
            out = model(X)
            loss = criterion(out,y)

            total_loss += loss.item()
            correct += (out.argmax(1)==y).sum().item()
    return total_loss,correct/len(loader.dataset)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=5e-4)

losses = []
val_losses = []
#学習
for epoch in range(30):
    running_loss = 0
    for inputs,labels in trainloader:
        inputs,labels = inputs.to(device),labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    val_loss,val_acc = evaluate(net,valloader)

    losses.append(running_loss)
    val_losses.append(val_loss)
    print(f'Epoch{epoch+1}')
    print(f'Train Loss:{running_loss:.3f}')
    print(f'Val Loss:{val_loss:.3f},Acc:{val_acc:.3f}')
#テスト
correct = 0
total = 0

with torch.no_grad():
    for images,labels in testloader:
        images,labels = images.to(device),labels.to(device)

        outputs = net(images)
        _,predicted = torch.max(outputs,1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy:{100*correct/total:2f}%')


import matplotlib.pyplot as plt
#plt.plot(losses,label='train')
#plt.plot(val_losses,label='val')
#plt.title("Training Loss")
#plt.show()

classes = train_data.dataset.classes if hasattr(train_data,"dataset") else train_data.classes

wrong_images = []
wrong_preds = []
wrong_labels = []

net.eval()

with torch.no_grad():
    for images,labels in testloader:
        images,labels = images.to(device),labels.to(device)

        outputs = net(images)
        _,predicted = torch.max(outputs,1)

        for i in range(len(labels)):
            if predicted[i] != labels[i]:
                wrong_images.append(images[i].cpu())
                wrong_preds.append(predicted[i].cpu())
                wrong_labels.append(labels[i].cpu())
        
        if len(wrong_images) >= 10:
            break
plt.figure(figsize=(10,5))

for i in range(10):
    img = wrong_images[i]

    img = img*0.5 + 0.5
    img = img.permute(1,2,0)

    plt.subplot(2,5,i+1)
    plt.imshow(img)
    plt.title(f'P:{classes[wrong_preds[i]]}\nT:{classes[wrong_labels[i]]}')
    plt.axis('off')
plt.show()