import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import gzip
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(28, 128, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3)
        self.fc1 = nn.Linear(576, 64)  
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x): 
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 576)  
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x

class MNISTDataset(Dataset):
    def __init__(self, data_folder, data_name, label_name, transform=None):
        self.data_folder = data_folder
        self.data_name = data_name
        self.label_name = label_name
        self.transform = transform
        self.data, self.labels = self.load_data()

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)

    def load_data(self):
        data_path = f"{self.data_folder}/{self.data_name}"
        label_path = f"{self.data_folder}/{self.label_name}"
        with gzip.open(label_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        with gzip.open(data_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(labels), 1, 28, 28)
        return images, labels

compose = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((20, 20)),  
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(20)
])

def get_data_loader(data_folder, is_train, batch_size=16, shuffle=True):
    data_name = 'train-images-idx3-ubyte.gz' if is_train else 't10k-images-idx3-ubyte.gz'
    label_name = 'train-labels-idx1-ubyte.gz' if is_train else 't10k-labels-idx1-ubyte.gz'
    dataset = MNISTDataset(data_folder, data_name, label_name, transform=compose)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for x, y in test_data:
            output = net(x.to(device))
            pred = torch.argmax(output, dim=1)
            n_correct += (pred == y.to(device)).sum().item()
            n_total += y.size(0)
    return n_correct / n_total

def main():
    train_data = get_data_loader('/home/mtftau-5/workplace/MNIST/MNIST/raw/', is_train=True)
    test_data = get_data_loader('/home/mtftau-5/workplace/MNIST/MNIST/raw/', is_train=False)
    net = Net().to(device)
    print('Initial accuracy:', evaluate(test_data, net))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    for epoch in range(100):
        for x, y in train_data:
            x, y = x.to(device), y.to(device)
            net.zero_grad()
            output = net(x)
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        print('Epoch', epoch, 'accuracy:', evaluate(test_data, net))

if __name__ == "__main__":
    main()