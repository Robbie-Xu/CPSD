import torch
import torch.nn as nn
from torch.nn.functional import multilabel_soft_margin_loss
import torch.nn.functional as F

class MyLoss(nn.Module):
    def __init__(self, beta):
        super(MyLoss, self).__init__()
        print('the hyper-parameter beta is %f' % beta)
        self.beta = beta

    def forward(self, score1, score2, truth):
        loss1 = multilabel_soft_margin_loss(score1, truth)
        loss2 = multilabel_soft_margin_loss(score2, truth)
        return loss1 + loss2


class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2)
        return loss


class multilabel_categorical_crossentropy(nn.Module):
    """
    softmax + cross entropy
    """
    def __init__(self):
        super(multilabel_categorical_crossentropy, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = K.zeros_like(y_pred[..., :1])
        y_pred_neg = K.concatenate([y_pred_neg, zeros], axis=-1)
        y_pred_pos = K.concatenate([y_pred_pos, zeros], axis=-1)
        neg_loss = K.logsumexp(y_pred_neg, axis=-1)
        pos_loss = K.logsumexp(y_pred_pos, axis=-1)
        return neg_loss + pos_loss