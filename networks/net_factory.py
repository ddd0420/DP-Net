from networks.unet import UNet, UNet_2d, UNet_Fusion
from networks.VNet import VNet
import torch.nn as nn


def net_factory(net_type="unet", in_chns=1, class_num=2, mode="train", tsne=0):
    if net_type == "unet" and mode == "train":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()

    if net_type == "unet_fusion" and mode == "train":
        net = UNet_Fusion(in_chns=in_chns, class_num=class_num).cuda()

    if net_type == "unet" and mode == "test":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()

    if net_type == "VNet" and mode == "train" and tsne == 0:
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    if net_type == "VNet" and mode == "test" and tsne == 0:
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    return net


def BCP_net(net_type="unet", in_chns=1, class_num=2, ema=False):
    if net_type == "unet":
        net = UNet_2d(in_chns=in_chns, class_num=class_num).cuda()
        if ema:
            for param in net.parameters():
                param.detach_()
    if net_type == "unet_fusion":
        net = UNet_Fusion(in_chns=in_chns, class_num=class_num).cuda()
        if ema:
            for param in net.parameters():
                param.detach_()

    return net

# def BCP_net(in_chns=1, class_num=2, ema=False):
#     net = UNet_2d(in_chns=in_chns, class_num=class_num).cuda()
#     if ema:
#         for param in net.parameters():
#             param.detach_()
#     return net

