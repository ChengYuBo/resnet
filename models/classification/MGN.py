#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    http://kazuto1011.github.io
# Date:   06 March 2019

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow import keras
from torch.nn import init

from . import model_zoo, modules
from .modules import _ConvBnReLU, _Flatten, _SeparableConv2d, _SepConvBnReLU
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

__all__ = ["MGN_v1"]

modules._BN_KWARGS["eps"] = 1e-3
modules._BN_KWARGS["momentum"] = 0.99
_N_MIDDLES = 8

import copy

import torch
from torch import nn

from torchvision.models.resnet import resnet50, Bottleneck


def make_model():
    return MGN()


class MGN(nn.Module):
    def __init__(self):
        super(MGN, self).__init__()

        resnet = resnet50(pretrained=True)  # resnet50网络

        self.backone = nn.Sequential(

            resnet.conv1,  # 64*112*112
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,  # 64*56*56
        )
        res_conv2 = nn.Sequential(*resnet.layer1[1:])  # 256*56*56
        res_conv3 = nn.Sequential(*resnet.layer1[1:], *resnet.layer2)  # 512*28*28
        res_conv4 = nn.Sequential(*resnet.layer1[1:], *resnet.layer2, *resnet.layer3)  # 1024*14*14

        res_g_conv3 = resnet.layer2  # p0 512*28*28
        res_g_conv4 = resnet.layer3  # p4 1024*14*14
        res_g_conv5 = resnet.layer4  # p1 2048*7*7

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))

        res_p_conv5.load_state_dict(resnet.layer4.state_dict())
        self.p0 = nn.Sequential(copy.deepcopy(res_conv2), copy.deepcopy(res_g_conv3),
                                nn.Conv2d(512, 2048, 1, bias=False), nn.BatchNorm2d(2048), nn.ReLU())
        self.p4 = nn.Sequential(copy.deepcopy(res_conv3), copy.deepcopy(res_g_conv4),
                                nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048), nn.ReLU())
        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        pool2d = nn.MaxPool2d

        self.maxpool_zg_p4 = pool2d(kernel_size=(24, 8))  # part-4??????????????
        self.maxpool_zg_p0 = pool2d(kernel_size=(48, 16))  # part-0
        self.maxpool_zg_p1 = pool2d(kernel_size=(12, 4))  # part-1
        self.maxpool_zg_p2 = pool2d(kernel_size=(24, 8))  # part-2
        self.maxpool_zg_p3 = pool2d(kernel_size=(24, 8))  # part-3
        self.maxpool_zp2 = pool2d(kernel_size=(12, 8))  # part-2 可以划分为2个
        self.maxpool_zp3 = pool2d(kernel_size=(8, 8))  # part-3 可以划分分3个
        reduction = nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU())

        self._init_reduction(reduction)
        self.reduction_0 = copy.deepcopy(reduction)
        # print("p1降维----》", self.reduction_0)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)
        self.reduction_7 = copy.deepcopy(reduction)

        # p0,p4
        self.reduction_p0 = copy.deepcopy(reduction)
        # print("p0降维----》", self.reduction_p0)
        self.reduction_p4 = copy.deepcopy(reduction)

        # fc softmax loss
        # self.fc_id_2048_0 = nn.Linear(2048, num_classes)
        self.fc_id_2048_0 = nn.Linear(256, 751)
        self.fc_id_2048_1 = nn.Linear(256, 751)
        self.fc_id_2048_2 = nn.Linear(256, 751)
        # p0 p4
        self.fc_id_2048_3 = nn.Linear(256, 751)
        # print("p0全连接------------->", self.fc_id_2048_3)
        self.fc_id_2048_4 = nn.Linear(256, 751)

        self.fc_id_256_1_0 = nn.Linear(256, 751)
        self.fc_id_256_1_1 = nn.Linear(256, 751)
        self.fc_id_256_2_0 = nn.Linear(256, 751)
        self.fc_id_256_2_1 = nn.Linear(256, 751)
        self.fc_id_256_2_2 = nn.Linear(256, 751)

        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)

        # p0 p4
        self._init_fc(self.fc_id_2048_3)
        # print("debug2222222222")
        self._init_fc(self.fc_id_2048_4)

        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        # 批量归一化
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')  # 权值初始化
        # nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):

        x = self.backone(x)

        p1 = self.p1(x)
        # print('p1----------->>>>>', p1.size()) 2048*12*4
        p2 = self.p2(x)
        # print('p2----------->>>>>', p2.size()) 2048*24*8
        p3 = self.p3(x)
        # print('p3----------->>>>>', p3.size()) 2048*24*8

        p0 = self.p0(x)
        # print('p0----------->>>>>', p0.size()) 2048*48*16
        p4 = self.p4(x)
        # print('p4----------->>>>>', p4.size()) 2048*24*8

        zg_p1 = self.maxpool_zg_p1(p1)  # 1*1*2048
        # print('zg-p1------------------------>', zg_p1.size()) 2048*1*1
        zg_p2 = self.maxpool_zg_p2(p2)  # 2*1*2048
        # print('zg-p2------------------------>', zg_p2.size())
        zg_p3 = self.maxpool_zg_p3(p3)  # 3*1*2048
        # print('zg-p3------------------------>', zg_p3.size())

        zg_p0 = self.maxpool_zg_p0(p0)  # 1*1*2048
        # print('zg-p0------------------------>', zg_p0.size())
        zg_p4 = self.maxpool_zg_p2(p4)  # 1*1*2048
        # print('zg-p4------------------------>', zg_p4.size())

        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]

        fg_p1 = self.reduction_0(zg_p1).squeeze(dim=3).squeeze(dim=2)
        # print("fg_p1----------->", fg_p1.size()) 16*256
        fg_p2 = self.reduction_1(zg_p2).squeeze(dim=3).squeeze(dim=2)
        # print("fg_p2----------->", fg_p2.size())
        fg_p3 = self.reduction_2(zg_p3).squeeze(dim=3).squeeze(dim=2)
        # print("fg_p3----------->", fg_p3.size())

        fg_p0 = self.reduction_p0(zg_p0).squeeze(dim=3).squeeze(dim=2)
        # print("fg_p0----------->", fg_p0.size())
        fg_p4 = self.reduction_p4(zg_p4).squeeze(dim=3).squeeze(dim=2)
        # print("fg_p4----------->", fg_p4.size())

        f0_p2 = self.reduction_3(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction_4(z1_p2).squeeze(dim=3).squeeze(dim=2)
        f0_p3 = self.reduction_5(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction_6(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction_7(z2_p3).squeeze(dim=3).squeeze(dim=2)

        '''
        l_p1 = self.fc_id_2048_0(zg_p1.squeeze(dim=3).squeeze(dim=2))
        l_p2 = self.fc_id_2048_1(zg_p2.squeeze(dim=3).squeeze(dim=2))
        l_p3 = self.fc_id_2048_2(zg_p3.squeeze(dim=3).squeeze(dim=2))
        '''
        l_p1 = self.fc_id_2048_0(fg_p1)
        l_p2 = self.fc_id_2048_1(fg_p2)
        l_p3 = self.fc_id_2048_2(fg_p3)

        l_p0 = self.fc_id_2048_3(fg_p0)
        l_p4 = self.fc_id_2048_4(fg_p4)

        l0_p2 = self.fc_id_256_1_0(f0_p2)
        l1_p2 = self.fc_id_256_1_1(f1_p2)
        l0_p3 = self.fc_id_256_2_0(f0_p3)
        l1_p3 = self.fc_id_256_2_1(f1_p3)
        l2_p3 = self.fc_id_256_2_2(f2_p3)

        predict = torch.cat([fg_p1, fg_p2, fg_p3, fg_p0, fg_p4, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)

        return predict, fg_p1, fg_p2, fg_p3, fg_p0, fg_p4, l_p1, l_p2, l_p3, l_p0, l_p4, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3


def add_attribute(cls):
    cls.pretrained_source = "Keras"
    cls.channels = "RGB"
    cls.image_shape = (299, 299)
    cls.mean = torch.tensor([127.5, 127.5, 127.5])
    cls.std = torch.tensor([255.0, 255.0, 255.0])
    return cls


def MGN_v1(n_classes=1000, pretrained=False, **kwargs):
    model = MGN()
    model.pretrained_source = "Keras"
    model.channels = "RGB"
    model.image_shape = (299, 299)
    model.mean = torch.tensor([127.5, 127.5, 127.5])
    model.std = torch.tensor([255.0, 255.0, 255.0])
    #
    if pretrained:
        state_dict = model_zoo.load_keras_xceptionv1(model_torch=model)
        model.load_state_dict(state_dict)
        model = add_attribute(model)
    return model


if __name__ == "__main__":
    model = MGN_v1(n_classes=1000)
    model.eval()
    model.load_from_keras()

    image = torch.randn(1, 3, 299, 299)

    print("[test]")
    print("input:", tuple(image.shape))
    print("logit:", tuple(model(image).shape))
