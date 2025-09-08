import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


class LowPassConvGenerator(nn.Module):
    def __init__(self, in_channel, num_k=2, ratio=8):
        super(LowPassConvGenerator, self).__init__()
        self.num_k = num_k
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1_list = nn.ModuleList(
            [nn.Conv2d(in_channel // num_k * 2, in_channel // ratio, 1, bias=False) for _ in range(num_k)])
        self.fc2_list = nn.ModuleList(
            [nn.Conv2d(in_channel // ratio, 9 * in_channel, 1, bias=False) for _ in range(num_k)])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.shape
        kernels = []

        for i in range(self.num_k):
            start_idx = i * (c // self.num_k)
            end_idx = (i + 1) * (c // self.num_k)
            x_i = x[:, start_idx:end_idx, :, :]

            avg_feature_x = self.avg_pool(x_i)  # b, c/num_k, 1, 1
            max_feature_x = self.max_pool(x_i)  # b, c/num_k, 1, 1

            fc1 = self.fc1_list[i]
            fc2 = self.fc2_list[i]

            combined_feature = torch.cat([avg_feature_x, max_feature_x], dim=1)
            kernel = fc2(self.relu(fc1(combined_feature)))  # (b, 9c, 1, 1)

            kernel = kernel.reshape(b, c, 9, 1)
            kernel = F.softmax(kernel, dim=2)
            kernel = kernel.reshape(b, c, 3, 3)[0].unsqueeze(1)  # (c/num_k, 1, 3, 3)

            kernels.append(kernel)
        final_kernel = torch.cat(kernels, dim=0)  # (num_k * c, 1, 3, 3)

        return final_kernel



class HighPassConvGenerator(nn.Module):
    def __init__(self, in_channel, num_k=2, kernel_size=3, ratio=8):
        super(HighPassConvGenerator, self).__init__()
        self.num_k = num_k
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1_list = nn.ModuleList(
            [nn.Conv2d(in_channel // num_k * 2, in_channel // ratio, 1, bias=False) for _ in range(num_k)])
        self.fc2_list = nn.ModuleList(
            [nn.Conv2d(in_channel // ratio, 9 * in_channel, 1, bias=False) for _ in range(num_k)])
        self.relu = nn.ReLU(inplace=True)

        self.identity_kernel = nn.Parameter(torch.tensor([[0, 0, 0],
                                                          [0, 1, 0],
                                                          [0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                                            requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.shape  # 64,64,64
        kernels = []

        for i in range(self.num_k):
            start_idx = i * (c // self.num_k)
            end_idx = (i + 1) * (c // self.num_k)
            x_i = x[:, start_idx:end_idx, :, :]

            avg_feature_x = self.avg_pool(x_i)  # b, c/num_k, 1, 1
            max_feature_x = self.max_pool(x_i)  # b, c/num_k, 1, 1
            fc1 = self.fc1_list[i]
            fc2 = self.fc2_list[i]

            combined_feature = torch.cat([avg_feature_x, max_feature_x], dim=1)
            kernel = fc2(self.relu(fc1(combined_feature)))  # (b, 9c, 1, 1)

            kernel = kernel.reshape(b, c, 9, 1)
            kernel = F.softmax(kernel, dim=2)
            kernel = kernel.reshape(b, c, 3, 3)[0].unsqueeze(1)  # (c/num_k, 1, 3, 3)

            kernels.append(self.identity_kernel - kernel)
        final_kernel = torch.cat(kernels, dim=0)  # (num_k * c, 1, 3, 3)

        return final_kernel


class Freq_Fusion(nn.Module):
    def __init__(self, c_high, c_low, c_out, num_k=2):
        super(Freq_Fusion, self).__init__()
        assert c_high == c_low * 2, 'c_high must be 2 * c_low'
        self.num_k = num_k
        self.c_high = c_high
        self.c_low = c_low
        self.c_out = c_out

        self.init_high_conv = nn.Conv2d(c_high, c_high // 2, 1, 1, padding='same')
        self.y_conv1 = nn.Conv2d(c_high, c_low, 1, 1)
        self.x_conv2 = nn.Conv2d(c_high, c_low, 1, 1)
        self.HPG_conv = nn.Sequential(nn.Conv2d(c_low, c_low, 7, 1, padding='same', groups=c_low),
                                      nn.BatchNorm2d(c_low),
                                      nn.ReLU(inplace=True))
        self.LPG_conv = nn.Sequential(nn.Conv2d(c_low, c_low, 3, 1, padding='same', groups=c_low),
                                      nn.BatchNorm2d(c_low),
                                      nn.ReLU(inplace=True))
        self.Cat_conv = nn.Sequential(
            nn.Conv2d(c_high + c_low, c_high + c_low, 3, 1, padding='same', groups=c_high + c_low),
            nn.BatchNorm2d(c_high + c_low),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_high + c_low, c_low, 1, 1, padding='same'),
            nn.BatchNorm2d(c_low),
            nn.ReLU(inplace=True)
        )
        self.res_conv = nn.Conv2d(c_low, c_low * 2, 1, 1, padding=0)
        self.HPG = HighPassConvGenerator(c_low, num_k=self.num_k)  # (c_low, 1, 3, 3)
        self.LPG = LowPassConvGenerator(c_low, num_k=self.num_k)  # (c_low, 1, 3, 3)

        self.x_conv_fhnorm = nn.BatchNorm2d(self.c_low * 2)
        self.y_conv_flnorm = nn.BatchNorm2d(self.c_low * 2)

        self.plus_conv = nn.Sequential(
            nn.Conv2d(c_low, c_out, 3, 1, padding='same'),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(c_low, c_out, 3, 1, padding='same'),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, 1, padding='same'),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_low, y_high):
        y_high_up = F.interpolate(y_high, scale_factor=2, mode='bilinear', align_corners=True)
        cat_fuse = torch.cat([x_low, y_high_up], dim=1)  # c_low+c_high,64,64
        cat_fuse = self.Cat_conv(cat_fuse)

        x_h = self.HPG_conv(x_low)
        x_l = self.LPG_conv(x_low)
        kernel_L = self.LPG(x_l)
        kernel_H = self.HPG(x_h)  # 64,1,3,3

        res = x_low
        x_low = F.conv2d(x_low, kernel_H, padding=1, groups=kernel_H.shape[0] // self.num_k)  # 64,64,64
        x_low = self.x_conv_fhnorm(x_low + self.res_conv(res))
        x_low = self.x_conv2(x_low)

        y_high_up = self.init_high_conv(y_high_up)
        y_high_up = F.conv2d(y_high_up, kernel_L, padding=1, groups=kernel_L.shape[0] // self.num_k)  # 64,64,64
        y_high_up = self.y_conv_flnorm(y_high_up)
        y_high_up = self.y_conv1(y_high_up)

        plus_fuse = x_low + y_high_up
        plus_fuse = self.plus_conv(plus_fuse)
        fuse = plus_fuse + cat_fuse
        fuse = self.out_conv(fuse)

        return fuse
