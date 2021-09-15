import torch.nn as nn

from .alexnet import *
from .resnet import *


def build_model(args):
    #  base model
    model_dict = {
        'resnet18': resnet18,
        'resnet50': resnet50,
        'alexnet': alexnet
    }

    model = model_dict[args.net](pretrained=True, progress=True)
    # fix FC layer
    num_classes = 345 if args.domainnet else 7
    if args.net == 'alexnet':
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    return model
