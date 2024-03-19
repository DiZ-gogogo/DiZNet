import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import random

# 定义函数，随机从0-end的一个序列中抽取size个不同的数
def random_num(size, end):
    range_ls = [i for i in range(end)]
    num_ls = []
    for i in range(size):
        num = random.choice(range_ls)
        range_ls.remove(num)
        num_ls.append(num)
    return num_ls


class ThresholdMap(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ThresholdMap, self).__init__()
        self.current_test_id = 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x), inplace=True)
        x = self.conv2(x)
        x = F.relu(self.bn2(x), inplace=True)
        x = self.conv3(x)

        x = F.relu(self.bn3(x), inplace=True)
        x = self.conv4(x)
        threshold_map = torch.sigmoid(x)

        # feature_map_cata = threshold_map
        #
        # print(feature_map_cata.shape)
        #
        # v = feature_map_cata.data.squeeze(0)
        #
        # v = v.cpu().numpy()
        #
        # print(v.shape)  # torch.Size([512, 28, 28])
        # self.current_test_id += 1
        # image_shape = v.shape[1]
        #
        # # 随机选取25个通道的特征图
        # channel_num = random_num(1, v.shape[0])
        # plt.figure(figsize=(20, 20))
        # for index, channel in enumerate(channel_num):
        #     ax = plt.subplot(1, 1, index + 1, )
        #     plt.imshow(v[channel, :, :], cmap="gray")  # 灰度图参数cmap="gray"
        #
        # path = r"/media/a204/742CCF0E2CCEC9F6/DiZ_gogogo/DeepLeaning/pan_pp_stable_to_power/E2E_CVTD/outputs/detail_map_ic15/{}/".format(
        #     str(self.current_test_id))
        #
        # if not os.path.exists(path):
        #     os.mkdir(path)
        #
        # plt.savefig(
        #     r"/media/a204/742CCF0E2CCEC9F6/DiZ_gogogo/DeepLeaning/pan_pp_stable_to_power/E2E_CVTD/outputs/detail_map_ic15/" + str(
        #         self.current_test_id) + "/" + str(
        #         image_shape) + "_feature.jpg",
        #     dpi=600)


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


if __name__ == '__main__':
    feature = torch.randn(7, 20, 736, 736)
    input_text = torch.randn(7, 1, 256, 256)

    threshold_map_model = ThresholdMap(in_channels=20, out_channels=1)

    model = AddThreshold(in_channels=1)

    threshold_map = threshold_map_model(feature)

    out = model(input_text, threshold_map)
    print(out.shape)
