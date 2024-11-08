import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import gzip
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa



divece= 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
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
    def __init__(self,data_fild,data_name,label_name,transform=None):
       self.data_fild=data_fild
       self.data_name=data_name
       self.label_name=label_name
       self.transform=transform
       self.data,self.labels=self.load_data()
    
    def __getitem__(self, index):
        img,lable=  self.data[index],self.labels[index]
        return img,lable
    def __len__(self):
        return len(self.data)
    
    def load_data(self):
        data_path = f"{self.data_fild}/{self.data_name}"
        label_path = f"{self.data_fild}/{self.label_name}"
        with gzip.open(label_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        with gzip.open(data_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(labels), 1, 28, 28)
        return images, labels\
            


class DataAugmenter:
    def __init__(self):
        # 设置数据增强序列
        self.seq = self.set_up_augmentation()

    def set_up_augmentation(self):
        # 定义数据增强的操作
        return iaa.Sequential([
            iaa.Crop(px=(1, 16), keep_size=False),  # 随机裁剪
            iaa.Fliplr(0.5),  # 50%的概率水平翻转
            iaa.GaussianBlur(sigma=(0, 3.0))  # 高斯模糊
        ])

    def augment_images(self, images):
        # 对输入的图像进行数据增强
        return self.seq(images=images)

# 使用 DataAugmenter 类
if __name__ == "__main__":
    # 初始化数据增强器
    augmenter = DataAugmenter()

    # 加载图像数据
    data_folder = '/home/mtftau-5/workplace/MNIST/MNIST/raw/'
    data_name = 'train-images-idx3-ubyte.gz'
    label_name = 'train-labels-idx1-ubyte.gz'
    dataset = MNISTDataset(data_folder, data_name, label_name, transform=transforms.ToTensor())
    images = dataset.data

    # 应用数据增强
    augmented_images = augmenter.augment_images(images)

    
def get_data_loader(data_folder, is_train, batch_size=100, shuffle=True):
    data_name = 'train-images-idx3-ubyte.gz' if is_train else 't10k-images-idx3-ubyte.gz'
    label_name = 'train-labels-idx1-ubyte.gz' if is_train else 't10k-labels-idx1-ubyte.gz'
    dataset = MNISTDataset(data_folder, data_name, label_name, transform=transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



