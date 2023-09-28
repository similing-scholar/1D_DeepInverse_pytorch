import torch
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd
import torchkeras

from model.load_dataset_interface import Basic1DDataset, load_dataset
from model.DeepInverse_net import DeepInverse_1D
from model.model_interface import KerasModel
from utils.evaluate_model_interface import plot_metric


# ------------------------------------ step 0/5 : 配置设备------------------------------------
USE_GPU = True  # 设置设备（CPU或GPU）
device = torch.device("cuda:0" if USE_GPU else "cpu")
dtype = torch.float32  # 将模型的权重参数转换为 float32 类型
print(f'using device:{device}, dtype:{dtype}')

# ------------------------------------ step 1/5 : 加载数据------------------------------------
# 定义数据集文件路径
mask_file = './dataset/mask[42,101].h5'
input_file = './dataset/1Dspectrum[10000,101].h5'
output_file = './dataset/1Dspectrum[10000,101].h5'
# 创建数据集
my_dataset = Basic1DDataset(mask_file, input_file, output_file)

# 定义划分比例和batch大小
train_ratio = 0.8
val_ratio = 0.2
batch_size = 128
# 加载数据集
train_loader, val_loader, test_loader = load_dataset(my_dataset, train_ratio, val_ratio, batch_size)

# ------------------------------------ step 2/5 : 创建网络------------------------------------
net = DeepInverse_1D(1, 1)  # 加载一个网络
net.initialize_weights()  # 初始化权值
net.to(device, dtype)  # 将网络移动到gpu/cpu，可以考虑.cuda()!!!

# ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------
model = KerasModel(device=device, dtype=dtype, net=net,
                   loss_fn=nn.MSELoss(),
                   optimizer=optim.Adam(net.parameters(), lr=0.01)
                   )
# model = torchkeras.KerasModel(net=net,
#                               loss_fn=nn.MSELoss(),
#                               optimizer=optim.Adam(net.parameters(), lr=0.01),
#                               )

# ------------------------------------ step 4/5 : 训练模型并保存 --------------------------------------------------
time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
ckpt_path = f'logs/{time}_checkpoint.pt'
model.fit(train_data=train_loader,
          val_data=val_loader,
          epochs=50,
          ckpt_path=ckpt_path,
          patience=5,
          monitor='val_loss',
          mode='min')  # 监控的指标为验证集上的损失函数，模式为最小化

# ------------------------------------ step 5/5 : 绘制训练曲线评估模型 --------------------------------------------------
df_history = pd.DataFrame(model.history)  # 将训练过程中的损失和指标数据保存为DataFrame格式

fig = plot_metric(df_history, "loss")
fig_path = f'logs/{time}_history.png'
fig.savefig(fig_path, dpi=300)
