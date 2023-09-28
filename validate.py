"""
使用训练数据集来做验证
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from model.load_dataset_interface import Basic1DDataset, load_dataset
from model.DeepInverse_net import DeepInverse_1D


# ------------------------------------ step 0/4 : 配置设备------------------------------------
USE_GPU = True  # 设置设备（CPU或GPU）
device = torch.device("cuda:0" if USE_GPU else "cpu")
dtype = torch.float32  # 将模型的权重参数转换为 float32 类型
print(f'using device:{device}, dtype:{dtype}')

# ------------------------------------ step 1/4 : 加载数据------------------------------------
# 定义数据集文件路径
mask_file = './dataset/mask[42,101].h5'
input_file = './dataset/1Dspectrum[10000,101].h5'
output_file = input_file
# 第2类：实例化数据集（包含输入和输出）
train_dataset = Basic1DDataset(mask_file, input_file, output_file)
train_loader, val_loader, _ = load_dataset(train_dataset, 0.99, 0.01, 1)

# ------------------------------------ step 2/4 : 加载网络------------------------------------
net = DeepInverse_1D(1, 1)  # 加载一个网络
net.to(device, dtype)  # 将网络移动到gpu/cpu

# ------------------------------------ step 3/4 : 加载模型权重 ------------------------------------
ckpt_path = 'logs/2023_09_28_10_44_48_checkpoint.pt'
# 方法2：因为预测只需要网络和权重，可以在net中使用torch的load_state_dict
model = net
model.load_state_dict(torch.load(ckpt_path))

# ------------------------------------ step 4/4 : 模型验证结果展示 ------------------------------------
with (torch.no_grad()):
    for batch in val_loader:
        feature, label = batch  # 对于train_dataset，batch有两个元素
        feature = feature.to(device, dtype)
        result = model.forward(feature)

        print(result.shape, result.view(-1).cpu().numpy().shape)  # (1, 1, 101) -> (101)
        result_flat = result[0][0].tolist()
        label_flat = label[0][0].tolist()

        plt.plot(result_flat, label='predict')  # 不传入x会根据y自动创建x轴
        plt.plot(label_flat, label='label')
        plt.legend()
        plt.show()