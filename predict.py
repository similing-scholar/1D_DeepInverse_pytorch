import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import torchkeras

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
# 实例化预测数据集
predict_dataset = Basic1DDataset(mask_file, input_file, encoder='predict')
# 模型需要传入batch维度，为了方便，先进行转换
predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size=1, shuffle=False, pin_memory=True)

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

# ------------------------------------ step 4/4 : 模型预测 ------------------------------------
with torch.no_grad():
    for batch in predict_loader:
        batch = batch.to(device, dtype)
        results = model.forward(batch)
        print(results.shape)
