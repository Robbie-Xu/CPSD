import torchvision.models as models
from torch.nn import Parameter
from utils.util import *
import torch
import torch.nn as nn
import model.vit_transformer as vtf

# global hyper-para
dim_inp = 1024
depth = 3
heads = 4
dim_head = 1024
dim_ff = 1024

class ResNet101TF(nn.Module):
    def __init__(self, model, model_params):
        super(ResNet101TF, self).__init__()
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

        # self.pooling = nn.MaxPool2d(14, 14)

        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

        self.transformer = vtf.Transformer(
            dim=dim_inp,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=dim_ff,
            dropout=0.1
        )

        self.fc = nn.Sequential(
            nn.Linear(2048, dim_inp)
        )

        self.label_emb = nn.Sequential(
            nn.Linear(300, dim_inp),
            nn.ReLU(),
        )

        self.classify = nn.Sequential(nn.Linear(dim_inp, 1))

    def forward(self, feature, attr):
        batch_size = feature.size(0)
        attr = attr[0]
        attr = self.label_emb(attr).unsqueeze(0).expand(batch_size, attr.size(0), dim_inp)

        feature = self.features(feature)
        feature = feature.view(feature.size(0), feature.size(1), -1).clone().transpose(-1, -2)
        feature = self.fc(feature)
        feature = torch.cat((feature, attr), 1)

        feature = self.transformer(feature)
        classify_feature = feature[:, -attr.size(1):, :]

        src_classes = self.classify(classify_feature).squeeze(2)
        return src_classes


def build_model(model_params):
    model = models.resnet101(pretrained=False)
    if model_params.pretrain != '':
        print("=> loading pretrained model '{}'".format(model_params.pretrain))
        model.load_state_dict(torch.load(model_params.pretrain))
    return ResNet101TF(model, model_params)
