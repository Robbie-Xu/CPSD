import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn
from model.avg_pool import FastAvgPool2d

class Resnet101(nn.Module):
    def __init__(self, model, model_params):
        super(Resnet101, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )

        # self.pooling = nn.AdaptiveMaxPool2d((1, 1))

        self.pooling = FastAvgPool2d(flatten=True)

        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

        self.classify = nn.Linear(2048, model_params.num_classes)

    def forward(self, feature):
        feature = self.features(feature)
        feature = self.pooling(feature)
        feature = torch.flatten(feature, 1)
        src_classes = self.classify(feature)
        return src_classes


def build_model(model_params):
    model = models.resnet101(pretrained=False)
    if model_params.pretrain != '':
        print("=> loading pretrained model '{}'".format(model_params.pretrain))
        model.load_state_dict(torch.load(model_params.pretrain))
    return Resnet101(model, model_params)
