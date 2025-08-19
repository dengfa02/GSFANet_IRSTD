from utils import wavelet
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale, requires_grad=True)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class PWD2d(nn.Module):  # 小波+卷积下采样核
    def __init__(self, in_channels, out_channels, wt_type='db1'):
        super(PWD2d, self).__init__()

        assert in_channels * 4 == out_channels

        self.in_channels = in_channels
        self.wt_filter, self.iwt_filter = wavelet.create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)

        self.para_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=2, stride=2, groups=in_channels),
            nn.BatchNorm2d(in_channels * 4),
            nn.ReLU())  # 分组2*2卷积
        self.ca = ChannelAttention(out_channels)

        self.wavelet_scale = _ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1)
        self.conv1x1 = nn.Conv2d(out_channels, in_channels * 2, kernel_size=1, stride=1)
        self.bnorm = nn.BatchNorm2d(in_channels * 2)
        self.act = nn.ReLU()

    def forward(self, x):
        curr_x_ll = x

        x = self.para_conv(x)
        ca = self.ca(x)
        x = x * ca

        curr_shape = curr_x_ll.shape
        if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
            curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
            curr_x_ll = F.pad(curr_x_ll, curr_pads)

        curr_x = wavelet.wavelet_transform(curr_x_ll, self.wt_filter)
        shape_x = curr_x.shape
        curr_x_wt = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])

        curr_x_wt = curr_x_wt * ca
        curr_x_wt = self.wavelet_scale(curr_x_wt)

        out = x + curr_x_wt
        out = self.act(self.bnorm(self.conv1x1(out)))

        return out
