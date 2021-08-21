#!/usr/bin/env python
# coding: utf-8
# %%
import nni
import argparse
import logging
import os
import random
import time
import warnings
import PIL
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from tensorboardX import SummaryWriter
from tqdm.notebook import tqdm
from torch.optim.lr_scheduler import StepLR


# %%
from utils_ViT import save_checkpoint, AverageMeter, ProgressMeter, adjust_learning_rate, accuracy, LabelSmoothingLoss, EMA

from vit_pytorch.vit import ViT
from vit_pytorch.local_vit import LocalViT


# %%
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument("--download-dir",
                    help="where to download datasets")
parser.add_argument('-a', '--arch', metavar='ARCH', default='vit',
                    help='model architecture (default: vit)')
parser.add_argument('--model-dir', default='/tmp', type=str)
parser.add_argument('--max-steps', default=None, type=int, 
                    help='maximum number of steps to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='(default: 64) this is the total ')
parser.add_argument('-e', '--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--seed', default=42, type=int,
                    help='the number of seed')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--request-from-nni', default=False, action='store_true')
parser.add_argument('--num-classes', default=2, type=int)

parser.add_argument('--optimizer', default='adam', type=str)
parser.add_argument('-g','--gamma', default=0.7, type=float)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--moving-average-decay', default=0.9999, type=float)
parser.add_argument('--dont-adjust-learning-rate', default=False, action='store_true')


# ViT parser_argument
parser.add_argument('-s','--image-size', default=300, type=int)
parser.add_argument('-p','--patch-size', default=30, type=int)
parser.add_argument('-pn','--patch-num', default=10, type=int)
parser.add_argument('-vd','--vt-dim', default=128, type=int)
parser.add_argument('-d','--vt-depth', default=6, type=int)
parser.add_argument('-vh','--vt-heads', default=6, type=int)
parser.add_argument('-mlpd','--mlp-dim', default=300, type=int)
parser.add_argument('-ch','--channel', default=3, type=int)


# %%
head = '[%(asctime)-15s] %(message)s'
logging.basicConfig(format=head)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
best_acc = 0
device = 'cuda'


# %%
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# %%
### 원하는 parameter 넣어주기
def main():
    args = parser.parse_args()
    
    if args.request_from_nni:
        import nni
        tuner_params = nni.get_next_parameter()
        logger.info(str(tuner_params))
        if "lr" in tuner_params:
            args.lr = tuner_params["lr"]
        else:
            args.lr = 3e-5
            
        if "gamma" in tuner_params:
            args.gamma = tuner_params["gamma"]
        else:
            args.gamma = 0.7
            
        if "image_size" in tuner_params:
            args.image_size = tuner_params["image_size"]
            
        if "patch_size" in tuner_params:
            args.patch_size = tuner_params["patch_size"]
            
        if "patch_num" in tuner_params:
            args.patch_num = tuner_params["patch_num"]
            if args.image_size % patch_size == 0:
                args.patch_size = args.image_size / args.patch_size
            else:
                print('Image dimensions must be divisible by the patch size.')
                
        if "vt_dim" in tuner_params:
            args.vt_dim = tuner_params["vt_dim"]
            
        if "vt_depth" in tuner_params:
            args.vt_depth = tuner_params["vt_depth"]
            
        if "vt_heads" in tuner_params:
            args.vt_heads = tuner_params['vt_heads']
            
        if "mlp_dim" in tuner_params:
            args.mlp_dim = tuner_params['mlp_dim']
            
        
        
        seed_everything(args.seed)
        
        logger.info(str(args))

        # demonstrate that intermediate result is actually sent
        ## initialize the result to zero
        nni.report_intermediate_result(0.)

        args.model_dir = os.environ["PWD"]

    

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    # Simply call main_worker function
    main_worker(args.gpu, args)


# %%
def main_worker(gpu, args):
    global best_acc
    args.gpu = gpu
    
    
    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    
    # create model
    if 'vit' == args.arch:
        if args.request_from_nni:
            model = ViT(image_size=args.image_size,
                        patch_size=args.patch_size,
                        num_classes=args.num_classes,
                        dim=args.vt_dim,
                        depth = args.vt_depth,
                        heads = args.vt_heads,
                        mlp_dim = args.mlp_dim,
                        channels=args.channel
                        )
            logger.info("=> Creating Vision Transformer with configurations from NNI")

        else:
            logger.info("you should use request_from_nni")
      
    else:
        logger.info(f"{args.arch} arch does not exit")



    
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.to('cuda')
        


    if args.moving_average_decay:
        logger.info("Using moving average decay = %.6f" % args.moving_average_decay)
        ema = EMA(args.moving_average_decay)
        for name, param in model.named_parameters():
            if param.requires_grad:
                ema.register(name, param.data)
    else:
        ema = None

        
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    

    if args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)

        
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)


    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # scheduler
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    else:
        raise NotImplementedError("Your requested optimizer '%s' is not found" % args.optimizer)

    

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            if args.gpu is not None:
                # best_acc may be from a checkpoint from a different GPU
                best_acc = best_acc.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))


    ## 사용중인 하드웨어에 가장 적합한 알고리즘(텐서 크기나 conv 연산에 맞게?) 
    ## 아마도 tensor의 크기나 gpu memory에 따라 효율적인 convolution 알고리즘이 서로 다른것으로 파악.
    cudnn.benchmark = True
    
    

    train_transforms = transforms.Compose([
    transforms.Resize((args.image_size,args.image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    val_transforms = transforms.Compose([
    transforms.Resize((args.image_size,args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    logger.info('Using image size %d' % args.image_size)
    
    
    dataset_dir = args.download_dir
    
    train_dataset = datasets.ImageFolder(dataset_dir+"/train", transform=train_transforms)
    val_dataset = datasets.ImageFolder(dataset_dir+"/test", transform=val_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=True)


    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        pin_memory=True)


    writer = SummaryWriter(os.path.join(args.model_dir, "summary"))


    for epoch in range(args.epochs):
        if not args.dont_adjust_learning_rate:
            adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        if not train(train_loader, writer, model, criterion, optimizer, ema, epoch, args):
            break
  
        # evaluate on validation set
        acc = validate(val_loader, writer, model, criterion, epoch, args)

        if args.request_from_nni:
            import nni
            nni.report_intermediate_result(acc)

        # remember best acc@1 and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'best_acc': best_acc,
            'optimizer': args.optimizer,
            'model': model,
        }, is_best, filename=os.path.join(args.model_dir, "checkpoint.pth.tar"))

    writer.close()

    try:
        if args.request_from_nni:
            import nni
            nni.report_final_result(acc)
            logger.info("Reported intermediate results to nni successfully")
    except NameError:
        logger.info("No accuracy reported")
        pass


# %%
def train(train_loader, writer, model, criterion, optimizer, ema, epoch, args, print_freq=30):
    forward_time = AverageMeter('Forward', ':6.3f')
    criterion_time = AverageMeter('Criterion', ':6.3f')
    backward_time = AverageMeter('Backward', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top = AverageMeter('Acc', ':6.2f')

    
    progress = ProgressMeter(logger, len(train_loader), data_time, forward_time, 
                             criterion_time, backward_time, losses,
                             top, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    logger.info("Epoch %d starts" % epoch)

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        end = time.time()

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)

        # measure forward time
        forward_time.update(time.time() - end)
        end = time.time()
            
        loss = criterion(output, target)


        # measure accuracy and record loss
        acc, _ = accuracy(output, target, topk=(1, args.num_classes))
        losses.update(loss.item(), images.size(0))
        top.update(acc[0], images.size(0))

        
        # measure criterion time
        criterion_time.update(time.time() - end)
        end = time.time()

        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        # measure elapsed time
        backward_time.update(time.time() - end)
        end = time.time()

        current_step = len(train_loader) * epoch + i

        if i % print_freq == 0:
            progress.print(i)
            writer.add_scalar("train/loss", losses.val, current_step)
            writer.add_scalar("train/acc", top.val, current_step)


        if args.max_steps is not None and current_step > args.max_steps:
            return False

    return True


# %%
def validate(val_loader, writer, model, criterion, epoch, args, print_freq=30):
    data_time = AverageMeter('Data', ':6.3f')
    batch_time = AverageMeter('Batch', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top = AverageMeter('Acc', ':6.2f')

    progress = ProgressMeter(logger, len(val_loader), data_time, 
                             batch_time, losses, top,
                             prefix='Test: ')


    # switch to evaluate mode
    model.eval()
    logger.info("Evaluation starts")

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            end = time.time()

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)


            acc, _ = accuracy(output, target, topk=(1, args.num_classes))
            losses.update(loss.item(), images.size(0))
            top.update(acc[0].item(), images.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                current_step = epoch * len(val_loader) + i
                progress.print(i)
                writer.add_scalar("val/loss", losses.val, current_step)
                writer.add_scalar("val/acc", top.val, current_step)


        logger.info(' * Acc {top.avg:.3f} '
                    .format(top=top))

    return top.avg


# %%
if __name__ == '__main__':
    logger.info("Process launched")
    main()
    logger.info("Process succesfully terminated")
