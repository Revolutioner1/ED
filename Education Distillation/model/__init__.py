from .resnet20 import resnet20,resnet20_layer2,resnet20_linear2,resnet20_layer3,resnet20_linear3
from .resnet110 import resnet110
model_dict = {
    'resnet20':resnet20,
    'resnet20_block2':resnet20_layer2,
    'resnet20_reference2':resnet20_linear2,
    'resnet20_block3':resnet20_layer3,
    'resnet20_reference3':resnet20_linear3,
    'resnet110':resnet110
}