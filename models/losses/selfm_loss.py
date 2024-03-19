import torch
from torch import nn

selfMLoss = nn.BCEWithLogitsLoss()


def selfm_loss(kernels, target, mask,reduce=True):


    batch_size = kernels.size(0)
    kernels = torch.sigmoid(kernels)
    kernels = kernels.contiguous().view(batch_size, -1)

    # texts = torch.sigmoid(texts)
    # texts = texts.contiguous().view(batch_size, -1)

    # mask = mask.contiguous().view(batch_size, -1).float()

    # kernels = kernels * mask
    # target = target * mask


    # print(kernels.shape)
    # print(target.shape)

    target=target.to(torch.float64)

    target=target.view(batch_size, -1)

    loss = selfMLoss(kernels,target)

    # print(loss)

    if reduce:
        loss = torch.mean(loss)
    return loss
