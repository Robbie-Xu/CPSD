import os
import argparse
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from helper_functions.helper_functions import mAP, CutoutPIL, ModelEma, add_weight_decay
from model.ASL_losses import AsymmetricLoss
from model.q2l_utils.ASL_losses_q2l import AsymmetricLossOptimized
from model.KL_loss import DistillKL

from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
import json
import torchvision.models as models

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--data', help='path to dataset', default='./data/')
parser.add_argument('--lr', default=1e-5, type=float)
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
# parser.add_argument('--model', type=str, help='name of supported model (choose from \"101(ResNet101), x(ResNeXt), q2l, 101tf(ResNet101+TF), tres(TResNet)\")')
parser.add_argument('--subnum', type=str, help='count of clusters(1,2,3,5,8...)')
parser.add_argument('--partition', type=str, help='number to distinguish different partitions')
parser.add_argument('--dataset', type=str, default='coco', help='coco or nuswide')
parser.add_argument('--model-root', default="./models", type=str)
parser.add_argument('--alpha', type=float, default=0.5, help='importance of co-ocurrence kd loss')
parser.add_argument('--metric', type=str, help='using kl or mse as loss metric')
parser.add_argument('-t', type=str, help='teacher model architecture (choose from \"101(ResNet101), x(ResNeXt), q2l, 101tf(ResNet101+TF), tres(TResNet)\")')
parser.add_argument('-s', type=str, help='student model architecture (choose from \"101(ResNet101), x(ResNeXt), q2l, 101tf(ResNet101+TF), tres(TResNet)\")')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume-ema', default='', type=str, metavar='PATH',
                    help='path to latest ema checkpoint (default: none)')
parser.add_argument('--resume-e', default=0, type=int, help='number of last epoch')


def load_teacher(path_list, args, model_t, num_class_list):
    #create model
    print('==> loading teacher model')
    models_dict=[]
    for i, eachpath in enumerate(path_list):
        eachpath = os.path.join(args.model_root, eachpath)
        args.num_classes = num_class_list[i]
        model =  model_t.build_model(args).cuda()
        if os.path.isfile(eachpath):
            print("=> loading checkpoint '{}'".format(eachpath))
            checkpoint = torch.load(eachpath)
            filtered_dict = {k.replace("module.",""): v for k, v in checkpoint.items()}
            model.load_state_dict(filtered_dict)
            models_dict.append(model)

    print('==> done')
    return models_dict

def main():
    args = parser.parse_args()
    args.do_bottleneck_head = False
    args.data = os.path.join(args.data,args.dataset)

    if args.dataset=='coco':
        from coco_cluster import COCO2014
        num_classes=80
        train_dataset = COCO2014(args.data, phase='train', inp_name='data/coco/coco_glove_word2vec.pkl',
                             transform=transforms.Compose([
                                 transforms.Resize((args.image_size, args.image_size)),
                                 CutoutPIL(cutout_factor=0.5),
                                 RandAugment(),
                                 transforms.ToTensor(),
                             ]))
        val_dataset = COCO2014(args.data, phase='val', inp_name='data/coco/coco_glove_word2vec.pkl',
                           transform=transforms.Compose([
                               transforms.Resize((args.image_size, args.image_size)),
                               transforms.ToTensor()
                           ]))
    elif args.dataset=='nuswide':
        from nuswide_cluster import NUSWIDEClassification
        num_classes=81
        train_dataset = NUSWIDEClassification(args.data, 'Train', inp_name='data/nuswide/nuswide_glove_word2vec.pkl',
                             transform=transforms.Compose([
                                 transforms.Resize((args.image_size, args.image_size)),
                                 CutoutPIL(cutout_factor=0.5),
                                 RandAugment(),
                                 transforms.ToTensor(),
                             ]))
        val_dataset = NUSWIDEClassification(args.data, 'Test', inp_name='data/nuswide/nuswide_glove_word2vec.pkl',
                           transform=transforms.Compose([
                               transforms.Resize((args.image_size, args.image_size)),
                               transforms.ToTensor()
                           ]))
    args.num_classes = num_classes
    '''
    config the root dir of teacher models (co-oc. and dis-oc. should be in one dir)
    '''
    # model_root = os.path.join(args.model_root,args.partition,args.dataset,"combine",'cl'+args.subnum)
    # args.model_root = model_root

    print(args.model_root)
    print(args)
    # Setup model
    print('creating model...')

    is_tf_model = False
    if args.t == 'x':
        from model import model_resnext as model_t
    elif args.t == '101':
        from model import model_101 as model_t
    elif args.t == 'tres':
        from model import model_tres as model_t
    elif args.t == 'q2l':
        from model import model_q2l as model_t

    if args.dataset == 'coco':
        version_set = 'idx'
        name_set = 'coco'
    elif args.dataset == 'nuswide':
        version_set = 'nus'
        name_set = 'nus'

    if args.s == 'x':
        from model import model_resnext as model_s
        args.pretrain = "data/semi_weakly_supervised_resnext50_32x4-72679e44.pth"
    elif args.s == '101':
        from model import model_101 as model_s
    elif args.s == '101tf':
        from model import model_trans_101 as model_s
        is_tf_model = True
    elif args.s == 'tres':
        from model import model_tres as model_s
        args.pretrain = "data/tresnet_l_448.pth"
    elif args.s == 'q2l':
        from model import model_q2l as model_s

    model_name = 'dis_' + name_set + '_combine'+'_'+args.t+'2'+args.s+'_partition'+args.partition+'_cl'+args.subnum+'_'+args.metric+'_'+str(args.alpha)[2]+str(round(1.0-args.alpha,2))[2]
    
    '''
        'cl' means 'cluster', which further means dis-ocurrence branch.
        'co' means 'co-ocurrence' branch.
    '''
    path_list_cl = []
    path_list_co = []
    # find the teacher models and fill into the list above
    for eachmodel in os.listdir(args.model_root):
        if eachmodel.find(args.t+'_partition')>0 and eachmodel.find('dis')<0:
            if eachmodel.find("cluster")>0:
                path_list_cl.append(eachmodel)
            elif eachmodel.find("coocurrence")>0:
                path_list_co.append(eachmodel)
    path_list_cl.sort()
    path_list_co.sort()
    print("cluster models: ", path_list_cl)
    print("coocurence models: ", path_list_co)

    version_cl = version_set+'_cluster'+args.subnum+'_'+args.partition+'.json'
    version_co = version_set+'_coocurrence'+args.subnum+'_'+args.partition+'.json'

    num_class_list_cl = []
    with open(os.path.join(args.seg,version_cl),'r') as f:
        idx_cl = json.load(f)
    for eachpart in idx_cl: 
        num_class_list_cl.append(len(eachpart))
    print("idx_cl:",idx_cl)

    num_class_list_co = []
    with open(os.path.join(args.seg,version_co),'r') as f:
        idx_co = json.load(f)
    for eachpart in idx_co: 
        num_class_list_co.append(len(eachpart))
    print("idx_co:", idx_co)
    
    
    model =  model_s.build_model(args).cuda()
    print("DataParallel")
    model_ema = model_s.build_model(args).cuda()
    model_t_list_cl = load_teacher(path_list_cl, args, model_t, num_class_list_cl)
    model_t_list_co = load_teacher(path_list_co, args, model_t, num_class_list_co)

    if args.resume and args.resume_ema:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            filtered_dict = {k.replace("module.",""): v for k, v in checkpoint.items()}
            model.load_state_dict(filtered_dict)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

        if os.path.isfile(args.resume_ema):
            print("=> loading ema checkpoint '{}'".format(args.resume_ema))
            checkpoint = torch.load(args.resume_ema)
            filtered_dict = {k.replace("module.",""): v for k, v in checkpoint.items()}
            model_ema.load_state_dict(filtered_dict)
        else:
            print("=> no ema checkpoint found at '{}'".format(args.resume_ema))
            return

    model = torch.nn.DataParallel(model)
    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # Actuall Training
    train_multi_label_coco(model, num_class_list_cl, num_class_list_co, model_t_list_cl, model_t_list_co, train_loader, val_loader, args.lr, args.batch_size, model_name, idx_cl, idx_co, args.model_root, is_tf_model, model_ema, args.resume_e, args.alpha)


def train_multi_label_coco(model, num_class_list_cl, num_class_list_co, model_t_list_cl, model_t_list_co, train_loader, val_loader, lr, bs, model_name, idx_cl, idx_co, model_root, is_tf_model, model_ema=None, resume_epoch=0, alpha=0):
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82
    if model_ema:
        ema.set(model_ema)

    # set optimizer 
    Epochs = 80
    Stop_epoch = 80
    weight_decay = 1e-4

    if model_name == 'q2l':
        criterion_cls = AsymmetricLossOptimized(
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
        criterion_cls = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
        parameters = add_weight_decay(model, weight_decay)
        optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    
    if model_name.find("mse")>-1:
        criterion_dis = torch.nn.MSELoss(reduce=True, size_average=False)
        print("using MSE")
    elif model_name.find("kl")>-1:
        criterion_dis = DistillKL(4)
        print("using KL")
    else:
        print("NO ASSIGNED LOSS FUNC, using MSE!")
        criterion_dis = torch.nn.MSELoss(reduce=True, size_average=False)

    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs, pct_start=0.1)
    if resume_epoch>0:
        for epoch in range(resume_epoch-1):
            for i in range(len(train_loader)):
                scheduler.step()

    highest_mAP = 0
    highest_epoch = 0
    trainInfoList = []
    scaler = GradScaler()

    if is_tf_model:
        attr_ = None
        for i, (inputData, target) in enumerate(train_loader):
            attr_ = inputData[2][0].cuda().half()
            break

        attrs_cl = []
        for i in range(len(num_class_list_cl)):
            idxkeys = list(idx_cl[i].keys())
            attr_part = torch.ones((num_class_list_cl[i],300))
            for j in range(num_class_list_cl[i]):
                attr_part[j,:] = attr_[int(idxkeys[j]),:]
            attrs_cl.append(attr_part)

        attrs_co = []
        for i in range(len(num_class_list_co)):
            idxkeys = list(idx_co[i].keys())
            attr_part = torch.ones((num_class_list_co[i],300))
            for j in range(num_class_list_co[i]):
                attr_part[j,:] = attr_[int(idxkeys[j]),:]
            attrs_co.append(attr_part)
            
    for epoch in range(resume_epoch, Epochs):
        print(model_name +" is training.")
        if epoch > Stop_epoch:
            break
        for i, (inputData, target) in enumerate(train_loader):
            input = inputData[0].cuda()
            attr = inputData[2].cuda().half()
            target[target == 0] = 1
            target[target == -1] = 0

            target = target.cuda() # (batch,3,num_classes)

            output_t_cl = None
            output_t_co = None
            with autocast():  # mixed precision
                with torch.no_grad():
                    for j,eachteacher in enumerate(model_t_list_cl):
                        if is_tf_model:
                            output_temp_cl = eachteacher(input, attrs_cl[j].expand(input.size(0), attrs_cl[j].size(0), 300).cuda()).float() 
                        else:
                            output_temp_cl = eachteacher(input).float()
                        if output_t_cl is None:
                            output_t_cl = output_temp_cl
                        else:
                            output_t_cl = torch.cat((output_t_cl, output_temp_cl), 1)  # b*c*d

                    for j,eachteacher in enumerate(model_t_list_co):
                        if is_tf_model:
                            output_temp_co = eachteacher(input, attrs_co[j].expand(input.size(0), attrs_co[j].size(0), 300).cuda()).float() 
                        else:
                            output_temp_co = eachteacher(input).float() 
                        if output_t_co is None:
                            output_t_co = output_temp_co
                        else:
                            output_t_co = torch.cat((output_t_co, output_temp_co), 1)  # b*c*d

            dis_target_cl = torch.ones_like(output_t_cl)
            dis_target_co = torch.ones_like(output_t_co)

            sort_idx_cl = []
            sort_idx_co = []
            for eachidxdic in idx_cl:
                sort_idx_cl += list(eachidxdic.keys())
            for j, each in enumerate(sort_idx_cl):
                dis_target_cl[:, int(each)] = output_t_cl[:, j]

            for eachidxdic in idx_co:
                sort_idx_co += list(eachidxdic.keys())
            for j, each in enumerate(sort_idx_co):
                dis_target_co[:, int(each)] = output_t_co[:, j]

            with autocast():  # mixed precision
                if is_tf_model:
                    output = model(input, attr).float()
                else:
                    output = model(input).float()  # sigmoid will be done in loss !
            
            loss_dis_cl = criterion_dis(output, dis_target_cl)
            loss_dis_co = criterion_dis(output, dis_target_co)
            loss_cls = criterion_cls(output, target)
            scale_a = alpha*0.5

            loss =  0.5*loss_cls+scale_a*loss_dis_co +round(0.5-scale_a,2)*loss_dis_cl
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
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss_cls: {:.1f}, Loss_dis_cl: {:.1f}, Loss_dis_co: {:.1f}'
                      .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0],
                              loss_cls.item()*0.5, loss_dis_cl.item()*round(0.5-scale_a,2), loss_dis_co.item()*scale_a))

        model.eval()
        mAP_score, name = validate_multi(val_loader, model, ema)
        model.train()
        if mAP_score > highest_mAP:
            try:
                os.remove(os.path.join(
                    model_root, model_name+'-{}-{}.ckpt'.format(highest_mAP,highest_epoch)))
                os.remove(os.path.join(model_root, model_name+'.ckpt'))
            except:
                pass
            highest_epoch = epoch
            highest_mAP = mAP_score
            try:
                if name == "regular":
                    torch.save(model.state_dict(), os.path.join(model_root, model_name+'-{}-{}.ckpt'.format(highest_mAP,epoch)))
                    torch.save(ema.state_dict(), os.path.join(model_root, model_name+'.ckpt'))
                elif name == "ema":
                    torch.save(ema.state_dict(), os.path.join(model_root, model_name+'-{}-{}.ckpt'.format(highest_mAP,epoch)))
                    torch.save(model.state_dict(), os.path.join(model_root, model_name+'.ckpt'))
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
        attr = inputData[2].cuda().half()
        target[target == 0] = 1
        target[target == -1] = 0

        # compute output
        with torch.no_grad():
            with autocast():
                if is_tf_model:
                    output_regular = Sig(model(input, attr))
                    output_ema = Sig(ema_model.module(input, attr))                    
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
    if mAP_score_regular >= mAP_score_ema:
        max_score = mAP_score_regular
        max_name = "regular"
    else:
        max_score = mAP_score_ema
        max_name = "ema"

    return max_score, max_name


if __name__ == '__main__':
    main()