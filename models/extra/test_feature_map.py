import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torchvision.models.feature_extraction import create_feature_extractor
from model import Net
import time
import os
from utils import plot_curve, plot_image, one_hot

# 网络初始化
net = Net()
net.load_state_dict(torch.load('checkpoints/model_param_9.pkl'))

print(net)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=1, shuffle=False)

transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

feature_extractor_layer = "conv1"

feature_extractor = create_feature_extractor(net, return_nodes={feature_extractor_layer: "output"})

# 数据测试,取出一个batch

if not os.path.exists("./feature_maps/" + feature_extractor_layer):
    os.makedirs("./feature_maps/" + feature_extractor_layer)

# test data
total_correct = 0

for x, y in test_loader:
    # x = x.reshape(x.size(0), -1)

    out_feature = feature_extractor(x)

    # 这里没有分通道可视化
    feature_map_picture = out_feature["output"][0].transpose(0, 1).sum(1).detach().numpy()
    # print("feature_map_picture", feature_map_picture.shape)
    plt.imshow(feature_map_picture)
    plt.savefig(
        "./feature_maps/{}/{}_truth_{}_{}.jpg".format(feature_extractor_layer, feature_extractor_layer, y.item(),
                                                      time.mktime(time.localtime())))

    out = net(x)
    # out:[b,10]  ===> pred

    # 预测值取最大的概率
    pred = out.argmax(dim=1)
    corrct = pred.eq(y).sum().float().item()
    total_correct += corrct

total_num = len(test_loader.dataset)
acc = total_correct / total_num
print("test acc", acc)

# 数据测试,取出一个batch

x, y = next(iter(test_loader))

print("y truth is :", y.item())
out = net(x)
pred = out.argmax(dim=1)
print("y predict is :", pred.item())

# plot_image(x, pred, "test samples")
