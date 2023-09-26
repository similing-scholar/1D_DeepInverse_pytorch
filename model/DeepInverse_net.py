import torch.nn as nn
from torchsummary import summary
import torch


class DeepInverse_1D(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DeepInverse_1D, self).__init__()
        self.conv_bundle = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv1d(32, output_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        output = self.conv_bundle(x)
        return output

    # 定义权值初始化
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


if __name__ == "__main__":
    # 创建一个一维信号输入输出都为1的示例模型
    model = DeepInverse_1D(1, 1)
    model.initialize_weights()  # 初始化权值
    model.to(torch.device("cuda:0"), dtype=torch.float32)  # 将网络移动到gpu
    # print(model)
    summary(model, (1, 101))
