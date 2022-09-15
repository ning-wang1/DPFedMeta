'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG16_meta_1': [64, 'M', 64, 'M', 64, 'M', 64, 'M'],
    'VGG16_meta': [48, 'M', 48, 'M', 48, 'M', 48, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, input_shape, num_filters, class_num=10):
        super(VGG, self).__init__()
        self.input_shape = input_shape
        self.class_num = class_num
        self.pool_stride = 2
        self.pool_kernel_size = 2
        self.conv_kernel_size = 3

        if num_filters==48:
            vgg_name = 'VGG16_meta'
        elif num_filters==64:
            vgg_name = 'VGG16_meta_1'
        else:
            vgg_name = 'VGG16'

        self.features = self._make_layers(cfg[vgg_name])

        linear_input_size = self.get_linear_input_size()
        self.classifier = nn.Linear(linear_input_size, self.class_num)

    def get_linear_input_size(self):
        x = torch.zeros(self.input_shape)
        out = self.features(x)
        out = out.view(out.size(0), -1)
        linear_input_size = out.shape[1]
        return linear_input_size

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.input_shape[1]
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_stride)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=self.conv_kernel_size, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x

        return nn.Sequential(*layers)

