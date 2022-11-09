import torch.nn as nn
import torch
from torchvision import models


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


class individual_model_vgg(nn.Module):
    def __init__(self, num_classes, choice_list, learngene):
        super(individual_model_vgg, self).__init__()
        self.num_classes = num_classes
        self.choice_list = choice_list
        self.choice_block = []
        self.kernel_list = [1, 3, 5, 7]
        channel = [3, 64, 128, 256, 512, 512, 512, 512, 512, 512]

        for idx, num_kernel in enumerate(self.choice_list):
            inp, oup = channel[idx], channel[idx + 1]
            kernel_size = self.kernel_list[num_kernel]
            self.choice_block.append(Choice_Block_fe(inp, oup, kernel=kernel_size, stride=1))
            self.choice_block.append(nn.MaxPool2d(2, 2))
        self.features = nn.Sequential(*self.choice_block)

        self.dis_stage0 = nn.Sequential(
            nn.Conv2d(out_channels=512, in_channels=64, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.dis_stage1 = nn.Sequential(
            nn.Conv2d(out_channels=512, in_channels=128, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((1, 1))

        )
        self.dis_stage2 = nn.Sequential(
            nn.Conv2d(out_channels=512, in_channels=256, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((1, 1))

        )
        self.dis_stage3 = nn.Sequential(
            nn.Conv2d(out_channels=512, in_channels=512, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((1, 1))

        )

        self.learngene_mianstream = learngene
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool_mainstream = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, self.num_classes)
        )

    def forward(self, x):
        x_stage0 = self.feature[0:2](x)
        out_stage0 = self.dis_stage0(x_stage0)

        x_stage1 = self.feature[2:4](x_stage0)
        out_stage1 = self.dis_stage1(x_stage1)

        x_stage2 = self.feature[4:6](x_stage1)
        out_stage2 = self.dis_stage2(x_stage2)

        x_stage3 = self.feature[6:8](x_stage2)
        out_stage3 = self.dis_stage3(x_stage3)

        x_mainstream = self.learngene_mianstream(x_stage3)
        x_mainstream = self.maxpool(x_mainstream)
        out_mainstream = x_mainstream

        x_mainstream = self.avgpool_mainstream(x_mainstream)
        x_mainstream = torch.flatten(x_mainstream, 1)
        output = self.classifier(x_mainstream)
        return out_stage0, out_stage1, out_stage2, out_stage3, out_mainstream, output


def ours_vgg(num_classes=5):
    vgg_19 = models.vgg19_bn(pretrained=False)
    num_features = vgg_19.classifier[6].in_features
    new_classifier = list(vgg_19.classifier.children())[:-1]
    new_classifier.extend([torch.nn.Linear(num_features, 5)])
    vgg_19.classifier = torch.nn.Sequential(*new_classifier)
    teacher_model = vgg_19

    learngene = get_learngene(teacher_model, 43, 51, requires_grad=True)
    return individual_model_vgg(num_classes=num_classes, choice_list=[2, 3, 3, 3], learngene=learngene)


def get_learngene(model, start, end, requires_grad=False):
    learngene = nn.Sequential(*list(model.features.children())[start:end+1])
    if not requires_grad:
        for name, param in learngene.named_parameters():
            param.requires_grad = False
    return learngene

