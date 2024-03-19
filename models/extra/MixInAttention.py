import torch
import torch.nn as nn
import torch.nn.functional as F

class MixAttention(nn.Module):
    def __init__(self, in_channels, r=8):
        super(MixAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels*2, in_channels // r, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // r, in_channels*2, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(in_channels*2, in_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels*2, in_channels, 1, bias=False)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        # Spatial Attention
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        # Channel Attention
        avg_out = F.avg_pool2d(out, out.size()[2:])
        max_out = F.max_pool2d(out, out.size()[2:])
        z = torch.cat([avg_out, max_out], dim=1)
        z = self.conv1(z)
        z = self.relu(z)
        z = self.conv2(z)
        z = self.sigmoid(z)
        out = out * z.expand_as(out)
        out = self.conv3(out)
        out += x
        return out


if __name__ == '__main__':
    model = MixAttention(in_channels=128)
    print(model)

    input = torch.randn(1, 128, 256, 256)
    out = model(input)
    print(out.shape)

#
####
#
#
#
# 在这个模块中，我们使用了一个具有 $r$ 个输出通道的 $1\times1$ 卷积层将输入 $x$ 分别转换为 Query、Key 和 Value。对于 Multi-Head Self-Attention 和 Global Attention，我们使用了两个注意力头。最后，我们通过全连接层将注意力输出向量转换为一个缩放系数，然后将其应用到原始输入上，得到最终输出。

# 值得注意的是，为了保证输出与输入特征大小一致，我们需要将 Global Attention 的输出结果变形为 $(B,C,1,1)$ 的形状，然后通过乘法运算将其应用到输入上。
# 同时，在进行 Multi-Head Self-Attention 时，我们需要将通道数 $C$ 拆分为 $C' \times h$，其中 $C'$ 是
# #
#
#
# #
#
