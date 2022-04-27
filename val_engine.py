import os
import shutil
import time
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn as nn
from utils.util import *
from utils.util import AveragePrecisionMeter
from torch.cuda.amp import GradScaler, autocast



class MultiLabelEngine():
    def __init__(self, thre):
        self.alpha = 0
        # hyper-parameters
        self.evaluation = True
        # measure mAP
        self.thre = thre
        print("thre:", self.thre)
        self.regular_ap_meter = AveragePrecisionMeter(threshold=self.thre, difficult_examples=False)
        
    def meter_reset(self):
        self.regular_ap_meter.reset()


    def meter_print(self):
        print("starting metric......")
        regular_ap = 100 * self.regular_ap_meter.value()
        regular_map = regular_ap.mean()

        OP, OR, OF1, CP, CR, CF1 = self.regular_ap_meter.overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.regular_ap_meter.overall_topk(3)
        print('=================================================>>>>>>> Experimental Results')
        print('mAP score: {map:.3f}'.format(map=regular_map))
        print('CP: {CP:.4f}\t'
              'CR: {CR:.4f}\t'
              'CF1: {CF1:.4f}'
              'OP: {OP:.4f}\t'
              'OR: {OR:.4f}\t'
              'OF1: {OF1:.4f}\t'.format(CP=CP, CR=CR,
                                      CF1=CF1, OP=OP, OR=OR, OF1=OF1))
        print('OP_3: {OP:.4f}\t'
              'OR_3: {OR:.4f}\t'
              'OF1_3: {OF1:.4f}\t'
              'CP_3: {CP:.4f}\t'
              'CR_3: {CR:.4f}\t'
              'CF1_3: {CF1:.4f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k,
                                        CR=CR_k, CF1=CF1_k))
        return regular_map

    def learning(self, model, val_loader, is_tf_model):
        if self.evaluation:
            model.eval()
            self.meter_reset()
            self.validate(model, val_loader, is_tf_model)
            _ = self.meter_print()
      
    def validate(self, model, val_loader, is_tf_model):
        print("starting validation")
        val_loader = tqdm(val_loader, desc='Test')
        for i, (inputData, target) in enumerate(val_loader):
            input = inputData[0].cuda()
            attr = inputData[2].cuda().half()
            target[target == 0] = 1
            target[target == -1] = 0

            # compute output
            with torch.no_grad():
                with autocast():
                    if is_tf_model:
                        output_regular = model(input, attr).float()
                    else:
                        output_regular = model(input).float()

            self.regular_ap_meter.add(output_regular.data, target)

