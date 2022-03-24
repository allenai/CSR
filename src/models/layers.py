from src.simulation import flow
import torch
import torch.nn as nn
import torch.nn.functional as F


def upshuffle(in_planes, out_planes, upscale_factor, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes * upscale_factor ** 2,
                  kernel_size=kernel_size, stride=stride, padding=padding),
        nn.PixelShuffle(upscale_factor),
        nn.ReLU()
    )


def upshuffle_norelu(in_planes, out_planes, upscale_factor, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes * upscale_factor ** 2,
                  kernel_size=kernel_size, stride=stride, padding=padding),
        nn.PixelShuffle(upscale_factor),
    )


def combine_1x1(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 1, 1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(),
    )


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, name=None):
        super(ResBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if downsample is None:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm_layer(planes),
            )
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicConvLayer(nn.Module):
    def __init__(self, inplane, planes):
        super(BasicConvLayer, self).__init__()
        self.conv = conv3x3(inplane, planes)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return self.relu(x)


class ResUpLayer(nn.Module):
    def __init__(self, inplanes, planes):
        """Upsample and then pass through resblock

        Args:
            inplanes (int): input number of channels
            planes (int): output number of channels
        """
        super(ResUpLayer, self).__init__()
        down_planes, up_planes = inplanes
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = conv3x3(up_planes, planes)
        self.block0 = ResBlock(down_planes + planes, planes)
        self.block1 = ResBlock(planes, planes)

    def forward(self, down_feature, up_feature):
        x = self.upsample(up_feature)
        x = self.conv(x)
        if down_feature is not None:
            x = torch.cat((down_feature, x), 1)
        x = self.block0(x)
        x = self.block1(x)

        return x

class MultiscaleHead(nn.Module):

    def __init__(self, out_planes):
        super(MultiscaleHead, self).__init__()

        self.resup4 = ResUpLayer((256, 512), 256)
        self.resup3 = ResUpLayer((128, 256 + out_planes), 128)
        self.resup2 = ResUpLayer((64, 128 + out_planes), 64)
        self.resup1 = ResUpLayer((64, 64 + out_planes), 64)
        self.resup0 = ResUpLayer((0, 64 + out_planes), 32)

        self.flow4 = nn.Conv2d(
            in_channels=256, out_channels=out_planes, kernel_size=3, padding=1)

        self.flow3 = nn.Conv2d(
            in_channels=128, out_channels=out_planes, kernel_size=3, padding=1)

        self.flow2 = nn.Conv2d(
            in_channels=64, out_channels=out_planes, kernel_size=3, padding=1)

        self.flow1 = nn.Conv2d(
            in_channels=64, out_channels=out_planes, kernel_size=3, padding=1)

        self.flow0 = nn.Conv2d(
            in_channels=32, out_channels=out_planes, kernel_size=1)

    def forward(self, enc1, enc2, enc3, enc4, b, multiscale):
        x = self.resup4(enc4, b)
        flow4 = self.flow4(x)
        x = self.resup3(enc3, torch.cat((x, flow4), 1))
        flow3 = self.flow3(x)
        x = self.resup2(enc2, torch.cat((x, flow3), 1))
        flow2 = self.flow2(x)
        x = self.resup1(enc1, torch.cat((x, flow2), 1))
        flow1 = self.flow1(x)
        x = self.resup0(None, torch.cat((x, flow1), 1))
        flow0 = self.flow0(x)

        if multiscale:
            return flow4, flow3, flow2, flow1, flow0
        else:
            return flow0,

class ResUpHead(nn.Module):

    def __init__(self, planes):
        super(ResUpHead, self).__init__()

        self.resup4 = ResUpLayer((256, 512), 256)
        self.resup3 = ResUpLayer((128, 256), 128)
        self.resup2 = ResUpLayer((64, 128), 64)
        self.resup1 = ResUpLayer((64, 64), 64)
        self.resup0 = ResUpLayer((0, 64), 32)
        self.conv_out = nn.Conv2d(
            in_channels=32, out_channels=planes, kernel_size=1)

    def forward(self, enc1, enc2, enc3, enc4, b):
        x = self.resup4(enc4, b)
        x = self.resup3(enc3, x)
        x = self.resup2(enc2, x)
        x = self.resup1(enc1, x)
        x = self.resup0(None, x)

        return self.conv_out(x)
