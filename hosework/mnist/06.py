# import  torchvision,torch
# import  torchvision.transforms as transforms
# from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt

# train = torchvision.datasets.MNIST(root='',train=True, 
#                                    transform= transforms.ToTensor())

# dataloader = DataLoader(train, batch_size=50,shuffle=True, num_workers=4)
# for step, (x, y) in enumerate(dataloader):
#     b_x = x.shape
#     b_y = y.shape
#     print ('Step: ', step, '| train_data的维度' ,b_x,'| train_target的维度',b_y)
# # 本来就有


import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import gzip
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)


    def forward(self, x):  
        x=torch.nn.functional.relu(self.conv1(x.view(-1,1,28,28)))
        x=self.pool(x)
        x=torch.nn.functional.relu(self.conv2(x))
        x=self.pool(x)
        x=torch.nn.functional.relu(self.fc1(x.view(-1,5*5*64)))
        x=torch.nn.functional.relu(self.fc2(x))
        x=torch.nn.functional.relu(self.fc3(x))
        x=torch.nn.functional.log_softmax(self.fc4(x), dim=1)
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
            # 应用转换，这些转换期望输入是PIL图像或NumPy数组
            img = Image.fromarray(img)  # 将NumPy数组转换为PIL图像
            img = self.transform(img)  # 应用转换
            img = np.array(img)  # 将PIL图像转换回NumPy数组
        return torch.from_numpy(img), torch.tensor(label)
    
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


#a构建路径n字典


def get_data_loader(data_folder, is_train, batch_size=100, shuffle=True):
    data_dict = {
        'data_name': 'train-images-idx3-ubyte.gz' if is_train else 't10k-images-idx3-ubyte.gz',
        'label_name': 'train-labels-idx1-ubyte.gz' if is_train else 't10k-labels-idx1-ubyte.gz'
    }
    data_path = os.path.join(data_folder, data_dict['data_name'])
    label_path = os.path.join(data_folder, data_dict['label_name'])
    
    data_name = data_dict['data_name']
    label_name = data_dict['label_name']

    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图片转换为Tensor
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化
    ])
    

    dataset = MNISTDataset(data_folder, data_name, label_name, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle),data_path ,label_path
 

def evaluate(test_data, net):
    n_correct=0
    n_total=0
    with torch.no_grad():
        for (x, y) in test_data:
            output = net.forward(x.view(-1, 28 * 28).to(device))
            for i, output in enumerate(output):
                if torch.argmax(output)==y[i]:#这个是我的，能看懂（参考了网上的框架，修改了运行的地方其实就是加了个to.device）
                    n_correct+=1
                n_total+=1
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

    # for n, (x, _) in enumerate(test_data):
    #     if n > 100:
    #         break
    #     predict = torch.argmax(net(x[0].view(-1, 28 * 28).to(device)))
    #     plt.figure(n)
    #     plt.imshow(x[0].view(28, 28), cmap='gray')
    #     plt.title('Prediction: ' + str(int(predict)))
    # plt.show()

if __name__ == "__main__":
    main()