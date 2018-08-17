import torch.nn as nn


class Conv1x1_BN(nn.Module):

    def __init__(self, in_channels, out_channels):
        super()
        self.in_channels = in_channels
        self.out_channel = out_channels

        self.net = self.get_network()

    def get_network(self):
        layers = []
        layers.append(nn.Conv2d(in_channels=self.in_channels,
                                out_channels=self.out_channels,
                                kernel_size=1,
                                bias=False))
        layers.append(nn.BatchNorm2d(num_features=self.out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Conv3x3_BN(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.net = self.get_network()

    def get_network(self):
        layers = []
        layers.append(nn.Conv2d(in_channels=self.in_channels,
                                out_channels=self.out_channels,
                                kernel_size=3,
                                stride=self.stride,
                                padding=1,
                                bias=False))
        layers.append(nn.BatchNorm2d(num_features=self.out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Block
    """

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.downsample = downsample


class BasicBlock(Block):
    """
    Basic residual block
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__(in_channels, out_channels, stride, downsample)

        self.conv1 = self.get_conv1()
        self.conv2 = self.get_conv2()
        self.relu = nn.ReLU(inplace=True)

    def get_conv1(self):
        layers = []

        layers.append(Conv3x3_BN(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 stride=self.stride))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def get_conv2(self):
        layers = []

        layers.append(Conv3x3_BN(in_channels=self.out_channels,
                                 out_channels=self.out_channels,
                                 stride=self.stride))

        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)

        if self.downsample is not None:
            x = self.downsample(x)

        y += x
        y = self.relu(y)

        return y


class BottleneckBlock(Block):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__(in_channels, out_channels, stride, downsample)

        self.conv1 = self.get_conv1()
        self.conv2 = self.get_conv2()
        self.conv3 = self.get_conv3()
        self.ReLU = nn.ReLU(inplace=True)

    def get_conv1(self):
        layers = []

        layers.append(Conv1x1_BN(in_channels=self.in_channels,
                                 out_channels=self.out_channels))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def get_conv2(self):
        layers = []

        layers.append(Conv3x3_BN(in_channels=self.out_channels,
                                 out_channels=self.out_channels,
                                 stride=self.stride,
                                 downsample=self.downsample))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def get_conv3(self):
        layers = []

        layers.append(Conv1x1_BN(in_channels=self.out_channels,
                                 out_channels=self.out_channels *
                                 BottleneckBlock.expansion))

        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)

        if self.downsample is not None:
            x = self.downsample(x)

        y += x
        y = self.ReLU(y)

        return y


class ResNet(nn.Module):

    """ResNet Architecture"""

    def __init__(self, channels, class_count):
        super(ResNet, self).__init__()
        self.channels = channels
        self.class_count = class_count

        self.net = self.get_network()

    def get_network(self):
        """
        returns the structure of the network
        """
        layers = []
        in_channels = self.channels

        # conv1
        layers.append(nn.Conv2d(in_channels=in_channels,
                                out_channels=64,
                                kernel_size=7,
                                stride=2,
                                padding=3,
                                bias=False))
        layers.append(nn.BatchNorm2d(num_features=64))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        pass  # TODO:
