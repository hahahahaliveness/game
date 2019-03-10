#!/usr/bin/env mdl
from common import config
import torch
import torch.nn as nn
import torch.nn.functional as func


def conv_bn(inp, oup, kernel_size, stride, padding, group, has_relu=True):
    if has_relu:
        return nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size, stride, padding=padding, groups=group),
            nn.BatchNorm2d(oup, eps=1e-9),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size, stride, padding=padding, groups=group),
            nn.BatchNorm2d(oup, eps=1e-9),
        )


class resnext_block(nn.Module):
    def __init__(self, channels_input, channels, stride, group, has_proj=False):
        super().__init__()
        bottleneck = channels // 4
        self.has_proj = has_proj
        self.stride = stride
        assert (bottleneck % group == 0) and (bottleneck / group) % 4 == 0, (bottleneck, group)

        self.avgpool = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2))
        self.conv_shortcut = conv_bn(channels_input, channels, 1, 1, 0, 1, has_relu=False)
        self.conv_1x1_shrink = conv_bn(channels_input, bottleneck, 1, 1, 0, 1, has_relu=True)
        self.conv_3 = conv_bn(bottleneck, bottleneck, 3, stride, 1, group, has_relu=True)
        self.conv_1x1_expand = conv_bn(bottleneck, channels, 1, 1, 0, 1, has_relu=False)
        self.relu = nn.Sequential(nn.ReLU(inplace=True))

    def forward(self, x):
        proj = x
        if self.has_proj:
            if self.stride == 2:
                proj = self.avgpool(proj)
            proj = self.conv_shortcut(proj)

        x = self.conv_1x1_shrink(x)
        x = self.conv_3(x)
        x = self.conv_1x1_expand(x)

        x = x + proj
        x = self.relu(x)

        return x


class resnext_group(nn.Module):
    def __init__(self, stages, channels_input, channels, group, dsp=True):
        super().__init__()
        self.resnext_block_ops = nn.ModuleList()
        for i in range(stages):
            stride = 2 if (i == 0 and dsp) else 1
            has_proj = False if i > 0 else True
            if i == 0:
                self.resnext_block_ops.append(resnext_block(channels_input, channels, stride, group, has_proj))
            else:
                self.resnext_block_ops.append(resnext_block(channels, channels, stride, group, has_proj))

    def forward(self, x):
        for op in self.resnext_block_ops:
            x = op(x)

        return x


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        stages = [2, 2, 3, 3]
        N = [2, 6, 16, 16]
        #channels = [n*16 for n in N]
        channels = [2*16, 6*16, 16*16, 16*32]
        group = [n for n in N]

        self.conv_fir = conv_bn(3, 16, 5, 2, 2, 1, has_relu=False)
        self.relu = nn.Sequential(nn.ReLU(inplace=True))
        self.maxpool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.group2 = resnext_group(stages[0], 32, channels[0], group[0], dsp=False)
        self.group3 = resnext_group(stages[1], channels[0], channels[1], group[1])
        self.group4 = resnext_group(stages[2], channels[1], channels[2], group[2])
        self.group5 = resnext_group(stages[3], channels[2], channels[3], group[3])
        # self.newfc = nn.Sequential(nn.Linear(channels[-1], config.nr_class))
        self.dense_fc = conv_bn(channels[-1], config.nr_class, 1, 1, 0, 1, False)

    def forward(self, x):
        x1 = self.conv_fir(x)
        x = torch.cat((x1, -x1), dim=1)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        x = self.group5(x)
        # x = x.mean(dim=3).mean(dim=2)
        # x = self.newfc(x)
        # pred = func.softmax(x)
        # return x

        x = self.dense_fc(x)
        n, c = x.size()[0], x.size()[1]
        x_dense = x.reshape(n, c, -1).transpose_(1, 2).reshape(-1, c)
        dense_pred = func.softmax(x_dense, dim=1).reshape(n, -1, c)
        return dense_pred


if __name__ == "__main__":
    net = Network()
    print(net)
# vim: ts=4 sw=4 sts=4 expandtab
