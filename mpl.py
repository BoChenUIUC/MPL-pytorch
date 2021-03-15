import argparse
import logging
import math
import os
import random
import time

import numpy as np
import torch
from torch.cuda import amp
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm

from data import DATASET_GETTERS
from models import WideResNet, ModelEMA
from utils import (AverageMeter, accuracy, create_loss_fn,
                   save_checkpoint, reduce_tensor, model_load_state_dict)

logger = logging.getLogger(__name__)
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
        

def setup_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='CCVE', type=str, help='experiment name')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--data-path', default='./data', type=str, help='data path')
    parser.add_argument('--save-path', default='./checkpoint', type=str, help='save path')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'], help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000, help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true", help="expand labels to fit eval steps")
    parser.add_argument('--total-steps', default=300000, type=int, help='number of total steps to run')
    parser.add_argument('--eval-step', default=1000, type=int, help='number of eval steps to run')
    parser.add_argument('--start-step', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--workers', default=4, type=int, help='number of workers')
    parser.add_argument('--num-classes', default=10, type=int, help='number of classes')
    parser.add_argument('--dense-dropout', default=0, type=float, help='dropout on last dense layer')
    parser.add_argument('--resize', default=32, type=int, help='resize image')
    parser.add_argument('--batch-size', default=64, type=int, help='train batch size')
    parser.add_argument('--lr', default=0.01, type=float, help='train learning late')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD Momentum')
    parser.add_argument('--nesterov', action='store_true', help='use nesterov')
    parser.add_argument('--weight-decay', default=0, type=float, help='train weight decay')
    parser.add_argument('--ema', default=0, type=float, help='EMA decay rate')
    parser.add_argument('--warmup-steps', default=0, type=int, help='warmup steps')
    parser.add_argument('--student-wait-steps', default=0, type=int, help='warmup steps')
    parser.add_argument('--grad-clip', default=0., type=float, help='gradient norm clipping')
    parser.add_argument('--resume', default='', type=str, help='path to checkpoint')
    parser.add_argument('--evaluate', action='store_true', help='only evaluate model on validation set')
    parser.add_argument('--finetune', action='store_true',
                        help='only finetune model on labeled dataset')
    parser.add_argument('--finetune-epochs', default=125, type=int, help='finetune epochs')
    parser.add_argument('--finetune-batch-size', default=512, type=int, help='finetune batch size')
    parser.add_argument('--finetune-lr', default=1e-5, type=float, help='finetune learning late')
    parser.add_argument('--finetune-weight-decay', default=0, type=float, help='finetune weight decay')
    parser.add_argument('--finetune-momentum', default=0, type=float, help='finetune SGD Momentum')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
    parser.add_argument('--label-smoothing', default=0, type=float, help='label smoothing alpha')
    parser.add_argument('--mu', default=7, type=int, help='coefficient of unlabeled batch size')
    parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
    parser.add_argument('--temperature', default=1, type=float, help='pseudo label temperature')
    parser.add_argument('--lambda-u', default=1, type=float, help='coefficient of unlabeled loss')
    parser.add_argument('--uda-steps', default=1, type=float, help='warmup steps of lambda-u')
    parser.add_argument("--randaug", nargs="+", type=int, help="use it like this. --randaug 2 10")
    parser.add_argument("--amp", action="store_true", help="use 16-bit (mixed) precision")
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    args = parser.parse_args()
    return args

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)



def evaluate(args, test_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    test_iter = tqdm(test_loader, disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        end = time.time()
        for step, (images, targets) in enumerate(test_iter):
            data_time.update(time.time() - end)
            batch_size = targets.shape[0]
            if args.device != 'cpu':
                images = images.cuda()
                targets = targets.cuda()
            with amp.autocast(enabled=args.amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, (1, 5))
            losses.update(loss.item(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)
            batch_time.update(time.time() - end)
            end = time.time()
            test_iter.set_description(
                f"Test Iter: {step+1:3}/{len(test_loader):3}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. "
                f"top1: {top1.avg:.2f}. top5: {top5.avg:.2f}. ")

        test_iter.close()
        return losses.avg, top1.avg, top5.avg

def get_dataloader(args,train=True):
    transform_val = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_dataset = datasets.CIFAR10(args.data_path, train=train, 
                                    transform=transform_val, download=False)
    test_loader = DataLoader(test_dataset,
                             sampler=SequentialSampler(test_dataset),
                             batch_size=args.batch_size,
                             num_workers=args.workers)
    return test_loader

def get_model(args):
    if args.seed is not None:
        set_seed(args)

    if args.dataset == "cifar10":
        depth, widen_factor = 28, 2
    elif args.dataset == 'cifar100':
        depth, widen_factor = 28, 8

    student_model = WideResNet(num_classes=args.num_classes,
                               depth=depth,
                               widen_factor=widen_factor,
                               dropout=0,
                               dense_dropout=args.dense_dropout)

    if os.path.isfile(args.resume):
        print(f"=> loading checkpoint '{args.resume}'")
        loc = f'cuda:0'
        checkpoint = torch.load(args.resume, map_location=loc)
        if checkpoint['avg_state_dict'] is not None:
            model_load_state_dict(student_model, checkpoint['avg_state_dict'])
        else:
            model_load_state_dict(student_model, checkpoint['student_state_dict'])

        print(f"=> loaded checkpoint '{args.resume}' (step {checkpoint['step']})")
    else:
        print(f"=> no checkpoint found at '{args.resume}'")
        exit(1)

    if args.device != 'cpu':
        student_model.cuda()
    return student_model

def run_model(args, test_loader, model, datarange=None,TF=None,C_param=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    test_iter = tqdm(test_loader, disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        end = time.time()
        for step, (images, targets) in enumerate(test_iter):
            if step<datarange[0]:continue
            elif step>=datarange[1]:break
            # perform transformation
            if TF is not None:
                tf_imgs = None
                for th_img in images:
                    np_img = (th_img.permute(1,2,0).numpy()*255).astype(np.uint8)
                    tf_img = TF.transform(image=np_img, C_param=C_param)
                    tf_img = torch.from_numpy(tf_img/255).float().permute(2,0,1).unsqueeze(0)
                    if tf_imgs is None:
                        tf_imgs = tf_img
                    else:
                        tf_imgs = torch.cat((tf_imgs,tf_img),0)
                images = tf_imgs
            normalization = transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
            images = normalization(images)
            # end transformation
            data_time.update(time.time() - end)
            batch_size = targets.shape[0]
            if args.device != 'cpu':
                images = images.cuda()
                targets = targets.cuda()
            with amp.autocast(enabled=args.amp):
                outputs = model(images)

            acc1, acc5 = accuracy(outputs, targets, (1, 5))
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)
            batch_time.update(time.time() - end)
            end = time.time()
            test_iter.set_description(
                f"Test Iter: {step+1:3}/{len(test_loader):3}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. "
                f"top1: {top1.avg:.2f}. top5: {top5.avg:.2f}. ")

        test_iter.close()
        return top1.avg/100#, top5.avg

def run_model_multi_range(args, test_loader, model, ranges=None,TF=None,C_param=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    test_iter = tqdm(test_loader, disable=args.local_rank not in [-1, 0])
    acc,cr = [],[]
    with torch.no_grad():
        end = time.time()
        for step, (images, targets) in enumerate(test_iter):
            # perform transformation
            if TF is not None:
                tf_imgs = None
                for th_img in images:
                    np_img = (th_img.permute(1,2,0).numpy()*255).astype(np.uint8)
                    tf_img = TF.transform(image=np_img, C_param=C_param)
                    tf_img = torch.from_numpy(tf_img/255).float().permute(2,0,1).unsqueeze(0)
                    if tf_imgs is None:
                        tf_imgs = tf_img
                    else:
                        tf_imgs = torch.cat((tf_imgs,tf_img),0)
                images = tf_imgs
            normalization = transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
            images = normalization(images)
            # end transformation
            data_time.update(time.time() - end)
            batch_size = targets.shape[0]
            if args.device != 'cpu':
                images = images.cuda()
                targets = targets.cuda()
            with amp.autocast(enabled=args.amp):
                outputs = model(images)

            acc1, acc5 = accuracy(outputs, targets, (1, 5))
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)
            batch_time.update(time.time() - end)
            end = time.time()
            test_iter.set_description(
                f"Test Iter: {step+1:3}/{len(test_loader):3}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. "
                f"top1: {top1.avg:.2f}. top5: {top5.avg:.2f}. ")
            if (step+1) in ranges:
                acc += [float(top1.avg)/100]
                cr += [TF.get_compression_ratio() if TF is not None else 0]

        test_iter.close()
        return acc,cr

class Simulator:
    def __init__(self,train=True):
        self.opt = setup_opt()
        self.opt.resume = './checkpoint/cifar10-4K.5_best.pth.tar'
        self.model = get_model(self.opt)
        self.dataloader = get_dataloader(self.opt,train=train)
        self.num_batches = len(self.dataloader)

    def get_one_point(self, datarange, TF=None, C_param=None):
        # start counting the compressed size
        if TF is not None: TF.reset()
        acc = run_model(self.opt,self.dataloader,self.model,datarange,TF,C_param)
        # get the compression ratio
        cr = TF.get_compression_ratio() if TF is not None else 0
        return acc,cr

    def get_multi_point(self, ranges, TF=None, C_param=None):
        if TF is not None: TF.reset()
        acc,crs = run_model_multi_range(self.opt,self.dataloader,self.model,ranges,TF,C_param)
        return acc,crs

    def test(self):
        # dp = self.get_one_point((0,100))
        dp = self.get_multi_point([i*100 for i in range(1,8)])
        print (dp,self.num_batches)

if __name__ == '__main__':
    sim = Simulator(train=True)
    sim.test()
