import torch
from torch import nn
from torch.nn import functional as f
from models.resnet import *


class Conv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, use_refl=True):
        super(Conv3x3, self).__init__()
        padding = kernel_size // 2
        padding_mode = "reflect" if use_refl else "zeros"
        self.conv = nn.Conv2d(in_channel,
                              out_channel,
                              kernel_size,
                              padding=padding,
                              padding_mode=padding_mode)

    def forward(self, x):
        return self.conv(x)


class ConvELU(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, use_refl=True):
        super(ConvELU, self).__init__()
        self.conv = Conv3x3(in_channel, out_channel, kernel_size, use_refl)
        self.act = nn.ELU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class DepthDecoder(nn.Module):
    def __init__(self, inp_channels, scales=range(4), use_skips=True):
        super(DepthDecoder, self).__init__()
        self.num_ch_enc = inp_channels
        self.num_ch_dec = [16, 32, 64, 128, 256]
        self.scales = scales
        self.use_skips = use_skips
        self.convs = nn.ModuleDict()
        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs.add_module("upconv{:d}_{:d}".format(i, 0), ConvELU(num_ch_in, num_ch_out))
            num_ch_in = self.num_ch_dec[i]
            if use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            self.convs.add_module("upconv{:d}_{:d}".format(i, 1), ConvELU(num_ch_in, num_ch_out))
        for s in scales:
            self.convs.add_module("dispconv{:d}".format(s), nn.Sequential(Conv3x3(self.num_ch_dec[s], 1),
                                                                          nn.Sigmoid()))

    def forward(self, xs):
        outputs = list()
        x = xs[-1]
        for i in range(4, -1, -1):
            x = self.convs["upconv{:d}_{:d}".format(i, 0)](x)
            x = f.interpolate(x, scale_factor=2, mode="nearest")
            x = torch.cat([x, xs[i - 1]], dim=1) if self.use_skips and i > 0 else x
            x = self.convs["upconv{:d}_{:d}".format(i, 1)](x)
            if i in self.scales:
                outputs.append(self.convs["dispconv{:d}".format(i)](x))
        return outputs[::-1]


class PoseDecoder(nn.Module):
    def __init__(self, inp_channels):
        super(PoseDecoder, self).__init__()
        self.num_ch_enc = inp_channels
        self.convs = nn.Sequential(
            nn.Conv2d(self.num_ch_enc[-1], 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 6, 1)
        )
        self.avg = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, xs):
        x = self.convs(xs[-1])
        x = 0.01 * self.avg(x).flatten(1)
        axisangle = x[..., :3]
        translation = x[..., 3:]
        return axisangle, translation


class DepthNet(nn.Module):
    def __init__(self, scales=range(4)):
        super(DepthNet, self).__init__()
        self.encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.decoder = DepthDecoder(self.encoder.out_channels, scales=scales, use_skips=True)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class PoseNet(nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()
        self.encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1, input_images=2)
        self.decoder = PoseDecoder(self.encoder.out_channels)

    def forward(self, x):
        return self.decoder(self.encoder(x))
