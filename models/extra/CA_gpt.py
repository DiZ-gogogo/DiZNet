import torch
import torch.nn as nn
import torch.nn.functional as F


class CAModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CAModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()
        proj_query = self.conv1(x).view(b, -1, w * h).permute(0, 2, 1)
        proj_key = self.conv2(x).view(b, -1, w * h)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.conv3(x).view(b, -1, w * h)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        out = self.bn4(self.conv4(self.bn3(out)))
        out = self.gamma * out + x
        return out


if __name__ == '__main__':
    model = CAModule(in_channels=128, out_channels=128)
    print(model)

    input = torch.randn(3, 128, 256, 122)
    out = model(input)
    print(out.shape)
