import t,orch
import torch.nn as nn
import torch.nn.functional as F


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttention(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))

        self.max_pooling = nn.AdaptiveMaxPool2d((1, None))

        temp_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = h_swish()

        self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        short = x
        n, c, H, W = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        out = self.act1(self.bn1(self.conv1(x_cat)))
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv2(x_h))
        out_w = torch.sigmoid(self.conv3(x_w))

        return short * out_w * out_h


class ThresholdMap(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ThresholdMap, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x), inplace=True)
        x = self.conv4(x)
        threshold_map = torch.sigmoid(x)
        return threshold_map


class ThresholdedReLU(nn.Module):
    def __init__(self, threshold=0.1, inplace=False):
        super().__init__()
        self.threshold = threshold
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return F.threshold(x, self.threshold, 0, inplace=True)
        else:
            return F.threshold(x, self.threshold, 0)


class AddThreshold(nn.Module):
    def __init__(self, in_channels, threshold=0.1):
        super().__init__()
        self.threshold = threshold
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = ThresholdedReLU(self.threshold, inplace=True)

    def forward(self, x, threshold_map):
        threshold_map = self.relu(self.bn(self.conv(threshold_map)))
        return x + threshold_map


class ThresholdActivateCA(nn.Module):
    def __init__(self, in_channels, out_channels, threshold=0.1):
        super().__init__()

        self.add_t = AddThreshold(1, threshold)
        self.threashold_m = ThresholdMap(in_channels, out_channels)
        self.ca = CoordAttention(in_channels, out_channels)

    def forward(self, x):
        ca_out = self.ca(x)
        threashold_m_out = self.threashold_m(x)
        out = self.add_t(ca_out, threashold_m_out)
        return out


if __name__ == '__main__':


    set_channels = 12

    model = ThresholdActivateCA(in_channels=set_channels, out_channels=set_channels)
    print(model)

    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("总参数数量和：" + str(k))

    input = torch.randn(3, set_channels, 122, 125)
    out = model(input)
    print(out.shape)
