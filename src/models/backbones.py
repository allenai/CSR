from collections import OrderedDict

import torch
import torch.nn as nn
from src.models.imagenet_resnet import resnet18, resnet34, resnet50


def get_torchvision_model_class(class_str: str):
    if class_str == 'resnet18':

        return resnet18
    elif class_str == 'resnet34':

        return resnet34
    elif class_str == 'resnet50':

        return resnet50
    else:
        raise NotImplementedError('Only supports resnet 18, 34, 50 for now.')

class FeatureLearner(nn.Module):

    def __init__(
        self,
        in_channels=3,
        channel_width=64,
        pretrained=False,
        num_classes=0,
        backbone_str='resnet18'
    ):
        super().__init__()

        model_class = get_torchvision_model_class(backbone_str)

        self.resnet = None
        self.num_classes = num_classes

        if num_classes == 0:
            # case where we do not want the fc, only the resnet features
            self.resnet = model_class(pretrained=pretrained, width=channel_width)
            del self.resnet.fc
        else:
            # want the fc
            self.resnet = model_class(pretrained=pretrained, width=channel_width)

            # replace the fc if needed, do this in two steps as might want pretrained weights
            if num_classes != 1000:
                self.resnet.fc = nn.Sequential(OrderedDict([
                    ('fc', nn.Linear(self.resnet.fc.in_features, num_classes))
                ]))

        if in_channels != 3:
            # case where we do not have 3 chan input so we might need to replicate channels
            if pretrained:
                # copy RGB feature channels in that order as necessary until we have reinitialized resnet.conv1
                weight = self.resnet.conv1.weight.detach()
                self.resnet.conv1 = nn.Conv2d(
                    in_channels, channel_width, kernel_size=7, stride=2, padding=3, bias=False)
                self.resnet.conv1.weight.data[:, :3] = weight.clone()
                for i in range(3, self.resnet.conv1.weight.data.shape[1]):
                    rhs_i = i % 3
                    self.resnet.conv1.weight.data[:,
                                                  i] = weight[:, rhs_i].clone()
            else:
                self.resnet.conv1 = nn.Conv2d(
                    in_channels, channel_width, kernel_size=7, stride=2, padding=3, bias=False)

        # memory to save the resnet intermediate features
        self.intermediate_features = None

    def forward(self, x):
        result = []

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        result.append(x)

        x = self.resnet.maxpool(x)

        # layer 1
        x = self.resnet.layer1(x)
        result.append(x)

        # layer 2
        x = self.resnet.layer2(x)
        result.append(x)

        # layer 3
        x = self.resnet.layer3(x)
        result.append(x)

        # layer 4
        x = self.resnet.layer4(x)
        result.append(x)

        self.intermediate_features = result

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        if self.num_classes == 0:
            # case where fc does not exist
            return x

        return self.resnet.fc(x)

class FeedForward(nn.Module):

    def __init__(self, layer_sizes):
        super(FeedForward, self).__init__()
        layers = []
        for i in range(1, len(layer_sizes)-1):
            layers.append(nn.Linear(
                layer_sizes[i-1], layer_sizes[i])),
            layers.append(nn.BatchNorm1d(layer_sizes[i])),
            layers.append(nn.ReLU())
        layers.append(nn.Linear(
            layer_sizes[-2], layer_sizes[-1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        return self.layers(x)
