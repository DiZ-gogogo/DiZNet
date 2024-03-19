import torch
import torch.nn as nn
import torch.nn.functional as F


class ASFModule(nn.Module):
    def __init__(self, in_channels, out_channels, scale_sizes):
        super(ASFModule, self).__init__()
        self.scale_sizes = scale_sizes
        self.in_conv = nn.ModuleList()
        self.out_conv = nn.ModuleList()
        for size in scale_sizes:
            self.in_conv.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            self.out_conv.append(nn.Conv2d(out_channels, in_channels, kernel_size=1))
        self.softmax = nn.Softmax(dim=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 缩放特征图
        scale_features = [F.interpolate(x, size=s) for s in self.scale_sizes]
        # 特征融合
        fused_feature = 0
        for i, feature in enumerate(scale_features):
            feature = self.in_conv[i](feature)
            feature_weight = self.softmax(feature)
            fused_feature += feature_weight * feature
        # 自适应调节
        gamma = self.gamma.expand_as(fused_feature)
        scale_feature = gamma * fused_feature + x
        # 特征重建
        out_feature = 0
        for i, feature in enumerate(scale_features):
            feature_weight = self.softmax(self.out_conv[i](scale_feature))
            out_feature += feature_weight * feature
        return out_feature


if __name__ == '__main__':
    model = ASFModule(in_channels=128, out_channels=128, scale_sizes=[(0, 1), (1, 2), (2, 3), (0, 3)])
    print(model)

    input = torch.randn(1, 128, 256, 256)
    out = model(input)
    print(out.shape)
