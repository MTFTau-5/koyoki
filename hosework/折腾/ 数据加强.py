import cv2
from skimage.util import random_noise
import torch
import numpy as np

# 使用torch.rand函数生成一个0到1之间的随机数
random_number = torch.rand(1).item()  # 使用.item()将tensor转换为Python数值
print(random_number)

#------加噪声------#
def _random_noise(img):
    return random_noise(img, mode='gaussian', clip=True)

#-----调整高度-----#

def _change_height(img):
    # 从边缘分布中采样
    alpha = torch.rand(1).item() * (1.2 - 0.8) + 0.8  # 生成0.8到1.2之间的随机数
    black = np.zeros((28, 28), dtype=np.uint8)  # 确保black是正确的数据类型
    return cv2.addWeighted(img, alpha, black, 1 - alpha, 0)

