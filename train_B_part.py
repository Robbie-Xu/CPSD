import os
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, add_weight_decay
from model.ASL_losses import AsymmetricLoss
from model.q2l_utils.ASL_losses_q2l import AsymmetricLossOptimized
import json

from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
import torchvision.models as models

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--data', help='path to dataset', default='./data/')
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=448, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--thre', default=0.8, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=24, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--print-freq', '-p', default=64, type=int,
                    metavar='N', help='print frequency (default: 64)')

parser.add_argument('--pretrain', default='data/resnet101.pth', type=str)
parser.add_argument('--num-classes', default=80)
parser.add_argument('--seg', default='./data/', help='root path of partition files')
parser.add_argument('--model', type=str, help='name of supported model (choose from \"101(ResNet101), x(ResNeXt), q2l, 101tf(ResNet101+TF), tres(TResNet)\")')
parser.add_argument('--subnum', type=str, help='count of clusters(1,2,3,5,8...)')
parser.add_argument('--part', default='',type=int, help='number of submodels')
parser.add_argument('--partition', type=str, help='number to distinguish different partitions')
parser.add_argument('--typ', default='cluster',type=str, help='coocurrence(CPG) or cluster(DPG)')
parser.add_argument('--dataset', type=str, default='coco', help='coco or nuswide')


def main():
    args = parser.parse_args()
    args.do_bottleneck_head = False

    if args.dataset == 'coco':
        args.num_classes=80
        version_set = 'idx'
        name_set = 'coco'
        
    elif args.dataset == 'nuswide':
        args.num_classes=81
        version_set = 'nus'
        name_set = 'nus'

    args.data = os.path.join(args.data,args.dataset)

    # version means the name of partition file
    version = version_set+'_'+args.typ +args.subnum+'_'+args.partition+'.json'
    model_name = name_set+'_'+args.typ+'_'+args.model+'_partition'+args.partition+'_cl'+args.subnum+'_'

    
    # _seg datasets are used for give the inputs and targets for each cluster given args.part
    if args.dataset=='coco':
        from coco_cluster import COCO2014_seg
        train_dataset = COCO2014_seg(args.data, args.seg, phase='train', inp_name='data/coco/coco_glove_word2vec.pkl',
                             transform=transforms.Compose([
                                 transforms.Resize((args.image_size, args.image_size)),
                                 CutoutPIL(cutout_factor=0.5),
                                 RandAugment(),
                                 transforms.ToTensor(),
                             ]), part=args.part, version=version)
        val_dataset = COCO2014_seg(args.data, args.seg, phase='val', inp_name='data/coco/coco_glove_word2vec.pkl',
                           transform=transforms.Compose([
                               transforms.Resize((args.image_size, args.image_size)),
                               transforms.ToTensor()
                           ]), part=args.part, version=version)
    elif args.dataset=='nuswide':
        from nuswide_cluster import NUSWIDEClassification_seg
        train_dataset = NUSWIDEClassification_seg(args.data, 'Train', inp_name='data/nuswide/nuswide_glove_word2vec.pkl',
                             transform=transforms.Compose([
                                 transforms.Resize((args.image_size, args.image_size)),
                                 CutoutPIL(cutout_factor=0.5),
                                 RandAugment(),
                                 transforms.ToTensor(),
                             ]), seg = args.seg, part=args.part, version=version)
        val_dataset = NUSWIDEClassification_seg(args.data, 'Test', inp_name='data/nuswide/nuswide_glove_word2vec.pkl',
                           transform=transforms.Compose([
                               transforms.Resize((args.image_size, args.image_size)),
                               transforms.ToTensor()
                           ]), seg = args.seg , part=args.part, version=version)
    else:
        raise NotImplementedError('dataset'+args.dataset+ 'is not implemented, please check the spell.')

    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))
    f_j = None
    with open(os.path.join(args.seg,version),'r') as f:
        f_j = json.load(f)
    num_classes = len(f_j[args.part])
    args.num_classes = num_classes

    is_tf_model = False
    '''
    import your model and set the path of pretrain model
    '''
    if args.model == '101':
        from model import model_101 as model_b
    elif args.model == 'x':
        from model import model_resnext as model_b
        args.pretrain = "data/semi_weakly_supervised_resnext50_32x4-72679e44.pth"
    elif args.model == '101tf':
        from model import model_trans_101 as model_b
        is_tf_model = True
    elif args.model == 'tres':
        from model import model_tres as model_b
        args.pretrain = "data/tresnet_l_448.pth"
    elif args.model == 'q2l':
        from model import model_q2l as model_b

    print(args)
    # Setup model
    print('creating model...')
    model = model_b.build_model(args).cuda()
    model = torch.nn.DataParallel(model)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # Actuall Training
    train_multi_label_coco(model, train_loader, val_loader, args.lr, args.part, model_name, args.batch_size, is_tf_model)


def train_multi_label_coco(model, train_loader, val_loader, lr, part_num, model_name, bs, is_tf_model):
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    # set optimizer
    Epochs = 80
    Stop_epoch = 80
    if model_name == 'q2l':
        weight_decay = 1e-2
        # criterion = nn.MultiLabelSoftMarginLoss(reduction="sum")
        criterion = AsymmetricLossOptimized(
            gamma_neg=2, gamma_pos=0,
            clip=0.0,
            disable_torch_grad_focal_loss=True,
            eps=1e-5,
        )
        lr_mult = bs / 256
        param_dicts = [{"params": [p for n, p in model.module.named_parameters() if p.requires_grad]},]
        optimizer = getattr(torch.optim,'AdamW')(
                param_dicts,
                lr_mult * 1e-4,
                betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay
            )
    else:
        weight_decay = 1e-4
        # criterion = nn.MultiLabelSoftMarginLoss(reduction="sum")
        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
        parameters = add_weight_decay(model, weight_decay)
        optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.1)

    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()
    for epoch in range(Epochs):
        print(model_name+str(part_num)+' is training.')
        if epoch > Stop_epoch:
            break
        for i, (inputData, target) in enumerate(train_loader):
            input = inputData[0].cuda()
            target[target == 0] = 1
            target[target == -1] = 0
            attr = inputData[2].cuda().half()
            target = target.cuda()  # (batch,3,num_classes)

            # target = target.max(dim=1)[0]
            with autocast():  # mixed precision
                '''
                notice that the model_name should not contain 'tf' unless its a model with word embedding (attr) as input,
                or modify the judgment here.
                '''
                if is_tf_model:
                    output = model(input,attr).float()
                else:
                    output = model(input).float()  # sigmoid will be done in loss !
            loss = criterion(output, target)
            model.zero_grad()

            scaler.scale(loss).backward()
            # loss.backward()

            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()

            scheduler.step()

            ema.update(model)
            # store information
            if i % 100 == 0:
                trainInfoList.append([epoch, i, loss.item()])
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                      .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0],
                              loss.item()))

        model.eval()
        mAP_score, name = validate_multi(val_loader, model, ema, is_tf_model)
        model.train()
        if mAP_score > highest_mAP:
            try:
                os.remove(os.path.join(
                    'models/', model_name+str(part_num)+'-{}.ckpt'.format(highest_mAP)))
            except:
                pass
            highest_mAP = mAP_score
            try:
                if name == "regular":
                    torch.save(model.state_dict(), os.path.join(
                        'models/', model_name +str(part_num)+'-{}.ckpt'.format(highest_mAP)))
                elif name == "ema":
                    torch.save(ema.state_dict(), os.path.join(
                        'models/', model_name +str(part_num)+'-{}.ckpt'.format(highest_mAP)))
                else:
                    assert 1 == 2
            except:
                pass
        print('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score, highest_mAP))


def validate_multi(val_loader, model, ema_model, is_tf_model):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    for i, (inputData, target) in enumerate(val_loader):
        input = inputData[0].cuda()
        target[target == 0] = 1
        target[target == -1] = 0
        attr = inputData[2].cuda().half()
        # compute output
        with torch.no_grad():
            with autocast():
                if is_tf_model:
                    output_regular = Sig(model(input,attr))
                    output_ema = Sig(ema_model.module(input,attr))
                else:          
                    output_regular = Sig(model(input))
                    output_ema = Sig(ema_model.module(input))

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())

    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    print("mAP score EMA {:.2f}".format(mAP_score_ema))
    if mAP_score_regular >= mAP_score_ema:
        max_score = mAP_score_regular
        max_name = "regular"
    else:
        max_score = mAP_score_ema
        max_name = "ema"

    return max_score, max_name


if __name__ == '__main__':
    main()