import torch
import torch.nn as nn


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()

        # Spatial Attention Module
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # Channel Attention Module
        self.fc3 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc4 = nn.Linear(in_channels // reduction_ratio, in_channels)

    def forward(self, x):
        # Channel Attention
        chn_avg = x.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)  # [batch_size, in_channels, 1, 1]
        chn_avg = self.fc3(chn_avg.squeeze(-1).squeeze(-1))
        chn_avg = self.fc4(self.relu(chn_avg)).unsqueeze(-1).unsqueeze(-1)  # [batch_size, in_channels, 1, 1]

        chn_max = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        chn_max = self.fc3(chn_max.squeeze(-1).squeeze(-1))
        chn_max = self.fc4(self.relu(chn_max)).unsqueeze(-1).unsqueeze(-1)

        chn_att = self.sigmoid(chn_avg + chn_max)
        x = x * chn_att

        # Spatial Attention
        spa_avg = self.fc1(x)
        spa_avg = self.relu(spa_avg)
        spa_avg = self.fc2(spa_avg)
        spa_max = -self.fc1(-x)
        spa_max = self.relu(spa_max)
        spa_max = -self.fc2(-spa_max)

        spa_att = self.sigmoid(spa_avg + spa_max)
        x = x * spa_att

        return x


class CBAMConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1,
                 reduction_ratio=16):
        super(CBAMConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
        self.cbam = CBAM(in_channels, reduction_ratio)

    def forward(self, x):
        x = self.cbam(x)
        x = self.conv(x)
        return x


if __name__ == '__main__':
    model = CBAMConv2d(in_channels=256, kernel_size=3, out_channels=128, reduction_ratio=16)
    print(model)

    input = torch.randn(4, 256, 256, 256)
    out = model(input)
    print(out.shape)
