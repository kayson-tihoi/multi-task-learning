from .convnet import convnet4
from .resnet import resnet12
from .resnet import seresnet12
from .wresnet import wrn_28_10

from .resnet_new import resnet50
from .modified_resnet import modified_resnet18, modified_resnet34
from .modified_resnet_cifar import modified_resnet20, modified_resnet32


model_pool = [
    'convnet4',
    'resnet12',
    'seresnet12',
    'wrn_28_10',
]

model_dict = {
    'wrn_28_10': wrn_28_10,
    'convnet4': convnet4,
    'resnet12': resnet12,
    'seresnet12': seresnet12,
    'resnet50': resnet50,
    'modified_resnet18': modified_resnet18,
    'modified_resnet34': modified_resnet34,
    'modified_resnet20': modified_resnet20,
    'modified_resnet32': modified_resnet32,
}
