import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn
from model.avg_pool import FastAvgPool2d
import model.resnext as resnext
class ResNeXt(nn.Module):
    def __init__(self, model, model_params):
        super(ResNeXt, self).__init__()
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
    model = resnext.resnext50_32x4d_swsl(model_params.pretrain, progress=True)
    return ResNeXt(model, model_params)
