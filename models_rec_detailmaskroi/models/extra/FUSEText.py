import torch
import torch.nn as nn

class FAM(nn.Module):
    def __init__(self, in_channels):
        super(FAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 4, in_channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        return out

class TextFuseNet(nn.Module):
    def __init__(self, in_channels):
        super(TextFuseNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.fam1 = FAM(32)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)
        self.fam2 = FAM(128)

    def forward(self, x):
        out = self.conv1(x)
        out = self.fam1(out)
        out = self.conv2(out)
        out = self.fam2(out)
        return out
if __name__ == '__main__':
    model = TextFuseNet(in_channels=128)
    print(model)

    input = torch.randn(1, 128, 256, 256)
    out = model(input)
    print(out.shape)
