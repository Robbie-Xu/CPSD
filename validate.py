import os
import argparse
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from val_engine import MultiLabelEngine
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--data',  help='path to dataset', default='./data/')
parser.add_argument('--model', type=str)
parser.add_argument('--dataset', type=str, default='coco')
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

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrain', default='', type=str)

def main():
    args = parser.parse_args()
    args.do_bottleneck_head = False

    args.data = os.path.join(args.data,args.dataset)

    if args.dataset=='coco':
        from coco_cluster import COCO2014
        num_classes=80
        val_dataset = COCO2014(args.data, phase='val', inp_name='data/coco/coco_glove_word2vec.pkl',
                           transform=transforms.Compose([
                               transforms.Resize((args.image_size, args.image_size)),
                               transforms.ToTensor()
                           ]))
    elif args.dataset=='nuswide':
        from nuswide_cluster import NUSWIDEClassification
        num_classes=81
        val_dataset = NUSWIDEClassification(args.data, 'Test', inp_name='data/nuswide/nuswide_glove_word2vec.pkl',
                           transform=transforms.Compose([
                               transforms.Resize((args.image_size, args.image_size)),
                               transforms.ToTensor()
                           ]))

    # model_root = os.path.join(args.model_root,args.partition,args.dataset,args.typ,'cl'+args.subnum)
    # args.model_root = model_root 
    args.num_classes = num_classes
    is_tf_model = False
    if args.model == '101':
        from model import model_101 as model_b
    elif args.model == 'x':
        from model import model_resnext as model_b
    elif args.model == '101tf':
        from model import model_trans_101 as model_b
        is_tf_model = True
    elif args.model == 'tres':
        from model import model_tres as model_b
    elif args.model == 'q2l':
        from model import model_q2l as model_b

    print(args)
    # Setup model
    print('creating model...')
    model = model_b.build_model(args).cuda()
    # model = torch.nn.DataParallel(model)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            filtered_dict = {k.replace("module.",""): v for k, v in checkpoint.items()}
            model.load_state_dict(filtered_dict)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    print("len(val_dataset)): ", len(val_dataset))

    # Pytorch Data loader

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # Actuall Training
    validate_multi(val_loader, model, args.thre, is_tf_model)


def validate_multi(val_loader, model, thre, is_tf_model):
    MLE = MultiLabelEngine(thre)
    MLE.learning(model, val_loader, is_tf_model)



if __name__ == '__main__':
    main()