import torch.nn as nn
import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

# from public.path import pretrained_models_path
pretrained_models_path = None

import torch
import torch.nn as nn

__all__ = [
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
]

model_urls = {
    'resnet18':
        '{}/resnetforcifar/resnet18-cifar-acc78.41.pth'.format(
            pretrained_models_path),
    'resnet34':
        '{}/resnetforcifar/resnet34-cifar-acc78.84.pth'.format(
            pretrained_models_path),
    'resnet50':
        '{}/resnetforcifar/resnet50-cifar-acc77.88.pth'.format(
            pretrained_models_path),
    'resnet101':
        '{}/resnetforcifar/resnet101-cifar-acc80.16.pth'.format(
            pretrained_models_path),
    'resnet152':
        '{}/resnetforcifar/resnet152-cifar-acc80.99.pth'.format(
            pretrained_models_path),
}


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.residual_branch = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels * BasicBlock.expansion,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion))

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * BasicBlock.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion))

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_branch(x) +
                                     self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()
        self.residual_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels,
                      stride=stride,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels * BottleNeck.expansion,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * BottleNeck.expansion,
                          stride=stride,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
                )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_branch(x) +
                                     self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=80, inter_layer=False):
        super(ResNet, self).__init__()
        self.inter_layer = inter_layer
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage2 = self._make_layer(block, 64, layers[0], 1)
        self.stage3 = self._make_layer(block, 128, layers[1], 2)
        self.stage4 = self._make_layer(block, 256, layers[2], 2)
        self.stage5 = self._make_layer(block, 512, layers[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x0 = x

        if self.inter_layer:
            x1 = self.stage2(x)
            x2 = self.stage3(x1)
            x3 = self.stage4(x2)
            x4 = self.stage5(x3)
            x = self.avg_pool(x4)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return [x0, x1, x2, x3, x4, x]
        else:
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.stage5(x)
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    # only load state_dict()
    if pretrained:
        model.load_state_dict(
            torch.load(model_urls[arch], map_location=torch.device('cpu')))

    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', BottleNeck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet101', BottleNeck, [3, 4, 23, 3], pretrained,
                   progress, **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet152', BottleNeck, [3, 8, 36, 3], pretrained,
                   progress, **kwargs)

############################################################################################
class Choice_Block_fe(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, supernet=True):
        super(Choice_Block_fe, self).__init__()
        padding = kernel // 2
        if supernet:
            self.affine = False
        else:
            self.affine = True
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.cb_main = nn.Sequential(
            # pw: 不改变输出特征图的长宽
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(self.out_channels, affine=self.affine),
            nn.ReLU(inplace=True))

    def forward(self, x):
        y = self.cb_main(x)
        return y


class individual_model_resnet(nn.Module):
    def __init__(self, num_classes, choice_list, learngene):
        super(individual_model_resnet, self).__init__()
        self.num_classes = num_classes
        self.choice_list = choice_list
        self.learngene = learngene
        self.choice_block = []
        self.kernel_list = [1, 3, 5, 7]
        self.num_channel = [64, 256, 512, 1024, 2048]
        self.stride = [1, 1, 2, 2, 2]

        for i, num_kernel in enumerate(self.choice_list):
            if i == 0:
                inp, oup = 3, self.num_channel[i]
                self.choice_block.append(Choice_Block_fe(inp, oup, kernel=self.kernel_list[num_kernel], stride=self.stride[i]))
            else:
                inp, oup = self.num_channel[i-1], self.num_channel[i]
                self.choice_block.append(BasicBlock(inp, oup, kernel_size=self.kernel_list[num_kernel], stride=self.stride[i], padding=int((self.kernel_list[num_kernel]-1)/2)))

        self.feature = nn.Sequential(*self.choice_block)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2048, num_classes)

        self.dis_stage0 = nn.Sequential(
            nn.Conv2d(out_channels=2048, in_channels=64, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.dis_stage1 = nn.Sequential(
            nn.Conv2d(out_channels=2048, in_channels=256, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((1, 1))

        )
        self.dis_stage2 = nn.Sequential(
            nn.Conv2d(out_channels=2048, in_channels=512, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((1, 1))

        )
        self.dis_stage3 = nn.Sequential(
            nn.Conv2d(out_channels=2048, in_channels=1024, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((1, 1))

        )

    def forward(self, x):
        x_stage0 = self.feature[0](x)
        out_stage0 = self.dis_stage0(x_stage0)

        x_stage1 = self.feature[1](x_stage0)
        out_stage1 = self.dis_stage1(x_stage1)

        x_stage2 = self.feature[2](x_stage1)
        out_stage2 = self.dis_stage2(x_stage2)

        x_stage3 = self.feature[3](x_stage2)
        out_stage3 = self.dis_stage3(x_stage3)

        x_mainstream = self.learngene(x_stage3)
        x_mainstream = self.avgpool(x_mainstream)
        out_mainstream = x_mainstream

        output = self.flatten(x_mainstream)
        return out_stage0, out_stage1, out_stage2, out_stage3, out_mainstream, output


def ours_resnet(num_classes=5):
    teacher_model = resnet50()
    learngene = teacher_model.stage5
    return individual_model_resnet(num_classes=num_classes, choice_list=[1, 1, 1, 2], learngene=learngene)