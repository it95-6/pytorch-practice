from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader,random_split
import torch
import torch.nn as nn
import torch.optim as optim

train_transform = transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

full_data = datasets.CIFAR10(
    root = 'data_cifar10',
    download = True,
    train = True,
    transform = train_transform
)

test_data = datasets.CIFAR10(
    root = 'data_cifar10',
    train = False,
    download = False,
    transform = test_transform
)

train_size = int(0.8*len(full_data))
val_size = len(full_data) - train_size
train_data , val_data = random_split(full_data,[train_size,val_size])

val_data.dataset.transform = test_transform

train_loader = DataLoader(train_data,batch_size=64,shuffle=True)
val_loader = DataLoader(val_data,batch_size=64,shuffle=False)
test_loader = DataLoader(test_data,batch_size=64,shuffle=False)

device = torch.device("mps" if torch.backends.mps.is_available() else 'cpu')

model = models.resnet18(weights=None)

model.fc = nn.Linear(model.fc.in_features,10)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=1e-3)

def evaluate(model,loader):
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for X,y in loader:
            X,y = X.to(device),y.to(device)

            outputs = model(X)
            loss = criterion(outputs,y)

            total_loss += loss.item()
            correct += (outputs.argmax(1)==y).sum().item()
    
    avg_loss = total_loss/len(loader)
    acc = correct/len(loader.dataset)
    return avg_loss,acc

best_val_acc = 0.0
train_losses = []
val_losses = []

for epoch in range(5):
    model.train()
    running_loss = 0.0

    for X,y in train_loader:
        X,y = X.to(device),y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs,y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    train_loss = running_loss/len(train_loader)
    val_loss,val_acc = evaluate(model,val_loader)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(),'best_resnet18.pth')
    
    print(f'Epoch{epoch+1}')
    print(f'Train Loss:{train_loss:.4f}')
    print(f'Val Loss:{val_loss:.4f},Val Acc:{val_acc}')

model.load_state_dict(torch.load('best_resnet18.pth',map_location=device))
test_loss,test_acc = evaluate(model,test_loader)

print(f'Test Loss:{test_loss:.4f}')
print(f'Test Accuracy:{test_acc*100:.2f}%')