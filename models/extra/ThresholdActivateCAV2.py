import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import random
import os


# 定义函数，随机从0-end的一个序列中抽取size个不同的数
def random_num(size, end):
    range_ls = [i for i in range(end)]
    num_ls = []
    for i in range(size):
        num = random.choice(range_ls)
        range_ls.remove(num)
        num_ls.append(num)
    return num_ls


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

        self.yu_num = 0
        self.current_test_id = -1

        temp_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = h_swish()

        self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)

        self.th_relu = ThresholdedReLU(0.1, inplace=True)

    def forward(self, x):
        short = x
        n, c, H, W = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)

        out = self.act1(self.bn1(self.conv1(x_cat)))

        th_out = self.th_relu(out)

        # 激活层 相加得到输出
        out = out + th_out

        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv2(x_h))
        out_w = torch.sigmoid(self.conv3(x_w))
        #
        # if self.yu_num % 4 == 0:
        #     self.current_test_id += 1
        #
        # if self.current_test_id == 279:
        #
        #     feature_map_cata = short * out_w * out_h
        #
        #     print(feature_map_cata.shape)
        #
        #     v = feature_map_cata.data.squeeze(0)
        #
        #     v = v.cpu().numpy()
        #
        #     print(v.shape)  # torch.Size([512, 28, 28])
        #
        #     image_shape = v.shape[1]
        #
        #     # 随机选取25个通道的特征图
        #     channel_num = random_num(25, v.shape[0])
        #     plt.figure(figsize=(20, 20))
        #     for index, channel in enumerate(channel_num):
        #         ax = plt.subplot(5, 5, index + 1, )
        #         plt.imshow(v[channel, :, :], cmap="gray")  # 灰度图参数cmap="gray"
        #
        #
        #
        #     print(self.current_test_id)
        #
        #     path = r"/media/diz/DE6A5F8C6A5F6077/UbantuSysterm/ZHOUDI/pan_pp_stable_to_power/pan_pp_stable-main/teding_feature_maps/{}/".format(
        #         str(self.current_test_id))
        #
        #     if not os.path.exists(path):
        #         os.mkdir(path)
        #
        #     plt.savefig(
        #         r"/media/diz/DE6A5F8C6A5F6077/UbantuSysterm/ZHOUDI/pan_pp_stable_to_power/pan_pp_stable-main/teding_feature_maps/" + str(
        #             self.current_test_id) + "/" + str(
        #             image_shape) + "_feature.jpg",
        #         dpi=600)
        #
        # self.yu_num += 1

        return short * out_w * out_h


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


if __name__ == '__main__':

    set_channels = 128

    model = CoordAttention(in_channels=set_channels, out_channels=set_channels)
    print(model)

    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("总参数数量和：" + str(k))

    input = torch.randn(3, set_channels, 252, 125)
    out = model(input)
    print(out.shape)
