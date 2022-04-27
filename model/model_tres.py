import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn
from tresnet import TResnetL
from tresnet.layers.avg_pool import FastAvgPool2d

def build_model(model_params):
    model = TResnetL(model_params)
    if model_params.pretrain != '':
        print("=> loading pretrained model '{}'".format(model_params.pretrain))
        state = torch.load(model_params.pretrain, map_location='cpu')
        filtered_dict = {k: v for k, v in state['model'].items() if
                         (k in model.state_dict() and 'head.fc' not in k)}
        model.load_state_dict(filtered_dict, strict=False)
    return model
