"""
使用预测脚本，通常用两种形式的数据集：1. 仅包含输入数据的数据集；2. 包含输入数据和输出数据的数据集。
"""
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import torchkeras
import numpy as np
import h5py

from model.load_dataset_interface import Basic1DDataset, load_dataset
from model.DeepInverse_net import DeepInverse_1D
from model.model_interface import KerasModel


# ------------------------------------ step 0/4 : 配置设备------------------------------------
USE_GPU = True  # 设置设备（CPU或GPU）
device = torch.device("cuda:0" if USE_GPU else "cpu")
dtype = torch.float32  # 将模型的权重参数转换为 float32 类型
print(f'using device:{device}, dtype:{dtype}')

# ------------------------------------ step 1/4 : 加载数据------------------------------------
# 定义数据集文件路径
mask_file = './dataset/mask[42,101].h5'
input_file = './dataset/1Dspectrum[100,42].h5'
# 第1类：实例化预测数据集（仅包含输入）
predict_dataset = Basic1DDataset(mask_file, input_file, encoder='predict')
# 模型需要传入batch维度，为了方便，先进行转换
predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size=1, shuffle=False, pin_memory=True)

# 第2类：实例化数据集（包含输入和输出）
train_dataset = Basic1DDataset(mask_file, './dataset/1Dspectrum[10000,101].h5', './dataset/1Dspectrum[10000,101].h5')
train_loader, val_loader, _ = load_dataset(train_dataset, 0.99, 0.01, 1)

# ------------------------------------ step 2/4 : 加载网络------------------------------------
net = DeepInverse_1D(1, 1)  # 加载一个网络
net.to(device, dtype)  # 将网络移动到gpu/cpu

# ------------------------------------ step 3/4 : 加载模型权重 ------------------------------------
ckpt_path = 'logs/2023_09_25_23_40_19_checkpoint.pt'
# 方法1：使用自定义的模型类下面的加载函数
# model = torchkeras.KerasModel(net=net,
#                               loss_fn=nn.MSELoss(),
#                               optimizer=optim.Adam(net.parameters(), lr=0.01),
#                               )
# model.load_ckpt(ckpt_path)

# 方法2：因为预测只需要网络和权重，可以在net中使用torch的load_state_dict
model = net
model.load_state_dict(torch.load(ckpt_path))

# ------------------------------------ step 4/4 : 模型预测与结果保存 ------------------------------------
results_array = None  # 用于存储结果的 NumPy 数组
with (torch.no_grad()):
    for batch in predict_loader:
        feature = batch  # 对于predict_dataset，batch只有一个元素
        # feature, label = batch  # 对于train_dataset，batch有两个元素
        feature = feature.to(device, dtype)
        result = model.forward(feature)

        print(result.shape, result.view(1, -1).shape)  # (1, 1, 101) -> (1, 101)
        result_flat = result.view(1, -1).cpu().numpy()
        # 将预测结果添加到results_array中
        if results_array is None:
            results_array = result_flat
        else:
            results_array = np.concatenate((results_array, result_flat), axis=0)

# 将结果保存为 h5 文件
with h5py.File('results/result[100,101].h5', 'w') as h5_file:
    h5_file.create_dataset('data', data=results_array)
    print(f'{results_array.shape} saved to h5 file.')
