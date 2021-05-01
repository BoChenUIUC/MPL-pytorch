import argparse
import logging
import math
import os
import random
import time

import numpy as np
import torch
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
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training')
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
    parser.add_argument("--use_subsampling", default=1, type=int, help="subsample to extract context")
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
    sampler = SequentialSampler(test_dataset)# if train==True else RandomSampler(test_dataset)
    test_loader = DataLoader(test_dataset,
                             sampler=sampler,
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
        loc = f'cpu'
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
        return top5.avg/100#, top5.avg

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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.autograd import Variable

def plot(samples):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig

def deepcod_main():
    from compression.deepcod import DeepCOD, orthorgonal_regularizer, init_weights
    sim_train = Simulator(train=True)
    sim_test = Simulator(train=False,usemodel=False)

    # data
    train_loader = sim_train.dataloader
    test_loader = sim_test.dataloader
    args = sim_train.opt
    use_subsampling=args.use_subsampling

    # discriminator
    app_model = sim_train.model
    app_model.eval()

    # encoder+decoder
    PATH = 'backup/deepcod_soft_ss_c8.pth' if use_subsampling else 'backup/deepcod_soft_c8.pth'
    max_acc = 0
    gen_model = DeepCOD(use_subsampling=use_subsampling)
    gen_model.apply(init_weights)
    if args.device != 'cpu':
        gen_model = gen_model.cuda()

    criterion_mse = nn.MSELoss()
    optimizer_g = torch.optim.Adam(gen_model.parameters(), lr=0.0001, betas=(0,0.9))
    normalization = transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    
    for epoch in range(1,1001):
        # training
        top1 = AverageMeter()
        top5 = AverageMeter()
        loss = AverageMeter()
        rlcr = AverageMeter()
        gen_model.train()
        train_iter = tqdm(train_loader, disable=args.local_rank not in [-1, 0])
        thresh = torch.FloatTensor([0.1,0])#torch.rand(2)
        if args.device != 'cpu': thresh = thresh.cuda()
        for step, (images, targets) in enumerate(train_iter):
            if args.device != 'cpu':
                images = images.cuda()
                targets = targets.cuda()

            # generator update
            optimizer_g.zero_grad()
            if use_subsampling:
                recon,res = gen_model((images,thresh))
            else:
                recon,r = gen_model(images)
            recon_labels,recon_features = app_model(normalization(recon),True)
            _,origin_features = app_model(normalization(images),True)

            loss_g = orthorgonal_regularizer(gen_model.encoder.sample.weight,0.0001,args.device != 'cpu')
            for origin_feat,recon_feat in zip(origin_features,recon_features):
                loss_g += criterion_mse(origin_feat,recon_feat)
            if use_subsampling:
                esti_cr,real_cr,std = res
                # loss_g += 0.0001*esti_cr - 0.0001*std
            
            loss_g.backward()
            optimizer_g.step()
            
            loss.update(loss_g.cpu().item())
            acc1, acc5 = accuracy(recon_labels, targets, (1, 5))
            top1.update(acc1[0], targets.shape[0])
            top5.update(acc5[0], targets.shape[0])
            rlcr.update(real_cr if use_subsampling else r)
            if use_subsampling:
                train_iter.set_description(
                    f"Train: {epoch:3}. Thresh: {thresh.cpu().item()[0]:.3f},{thresh.cpu().item()[1]:.3f}.  "
                    f"top1: {top1.avg:.2f}. top5: {top5.avg:.2f}. "
                    f"loss: {loss.avg:.3f}. cr: {rlcr.avg:.5f}. "
                    )
            else:
                train_iter.set_description(
                    f"Train: {epoch:3}. "
                    f"top1: {top1.avg:.2f}. top5: {top5.avg:.2f}. "
                    f"loss: {loss.avg:.3f}. cr: {rlcr.avg:.5f}. "
                    )

        train_iter.close()

        # testing
        if epoch%5!=0:continue
        # need to choose some anchors
        # thresh = torch.FloatTensor([0.1,0.2])
        # if args.device != 'cpu': thresh = thresh.cuda()
        print('Save to', PATH,thresh)
        top1 = AverageMeter()
        top5 = AverageMeter()
        loss = AverageMeter()
        rlcr = AverageMeter()
        # gen_model.eval()
        test_iter = tqdm(test_loader, disable=args.local_rank not in [-1, 0])
        for step, (images, targets) in enumerate(test_iter):
            if args.device != 'cpu':
                images = images.cuda()
                targets = targets.cuda()

            # generator update
            if use_subsampling:
                recon,res = gen_model((images,thresh))
            else:
                recon,r = gen_model(images)
            recon_labels,recon_features = app_model(normalization(recon),True)
            _,origin_features = app_model(normalization(images),True)

            loss_g = orthorgonal_regularizer(gen_model.encoder.sample.weight,0.0001,args.device != 'cpu')
            for origin_feat,recon_feat in zip(origin_features,recon_features):
                loss_g += criterion_mse(origin_feat,recon_feat)
            if use_subsampling:
                _,real_cr,_ = res
                # loss_g += 0.0001*esti_cr - 0.0001*std

            loss.update(loss_g.cpu().item())
            acc1, acc5 = accuracy(recon_labels, targets, (1, 5))
            top1.update(acc1[0], targets.shape[0])
            top5.update(acc5[0], targets.shape[0])
            rlcr.update(real_cr if use_subsampling else r)
            if use_subsampling:
                test_iter.set_description(
                    f" Test: {epoch:3}. Thresh: {thresh.cpu().item()[0]:.3f},{thresh.cpu().item()[1]:.3f}. "
                    f"top1: {top1.avg:.2f}. top5: {top5.avg:.2f}. "
                    f"loss: {loss.avg:.3f}. cr: {rlcr.avg:.5f}. "
                    )
            else:
                test_iter.set_description(
                    f" Test: {epoch:3}. "
                    f"top1: {top1.avg:.2f}. top5: {top5.avg:.2f}. "
                    f"loss: {loss.avg:.3f}. cr: {rlcr.avg:.5f}. "
                    )

        test_iter.close()
        if top5.avg > max_acc:
            torch.save(gen_model.state_dict(), PATH)
            max_acc = top5.avg

def deepcod_validate():
    from compression.deepcod import DeepCOD, compute_gradient_penalty
    sim = Simulator(train=False)

    # data
    test_loader = sim.dataloader
    args = sim.opt

    # discriminator
    app_model = sim.model
    app_model.eval()

    # encoder+decoder
    PATH = 'backup/deepcod_soft_ss_c8.pth'
    max_acc = 0
    gen_model = DeepCOD()
    gen_model.load_state_dict(torch.load(PATH,map_location='cpu'))
    if args.device != 'cpu':
        gen_model = gen_model.cuda()

    normalization = transforms.Normalize(mean=cifar10_mean, std=cifar10_std)

    gen_model.eval()

    for th1 in range(11):
        for th2 in range(11):
            thresh = torch.FloatTensor([th1/10.0,th2/10.0])
            if args.device != 'cpu': thresh = thresh.cuda()
            top1 = AverageMeter()
            top5 = AverageMeter()
            cr = AverageMeter()
            test_iter = tqdm(test_loader)
            for step, (images, targets) in enumerate(test_iter):
                if args.device != 'cpu':
                    images = images.cuda()
                    targets = targets.cuda()

                # generator update
                recon,res = gen_model((images,thresh))
                recon_labels = app_model(normalization(recon))

                esti_cr,real_cr,_ = res
                acc1, acc5 = accuracy(recon_labels, targets, (1, 5))
                top1.update(acc1[0], targets.shape[0])
                top5.update(acc5[0], targets.shape[0])
                cr.update(real_cr)
                test_iter.set_description(
                    f"top1: {top1.avg:.2f}. top5: {top5.avg:.2f}. "
                    f"Thresh: {thresh.cpu().numpy()[0]:.3f},{thresh.cpu().numpy()[1]:.3f}. "
                    f"real: {real_cr:.4f}. esti: {esti_cr:.4f}")
            with open("acc.log", "a") as f:
                f.write(f"{top5.avg:.3f}\n")
            with open("real_cr.log", "a") as f:
                f.write(f"{cr.avg:.5f}\n")

            test_iter.close()

def deepcod_validate2():
    from compression.deepcod import DeepCOD, compute_gradient_penalty
    sim = Simulator(train=False)

    # data
    test_loader = sim.dataloader
    args = sim.opt

    # discriminator
    app_model = sim.model
    app_model.eval()

    # encoder+decoder
    PATH = 'backup/deepcod_soft_c8.pth'
    max_acc = 0
    gen_model = DeepCOD(use_subsampling=False)
    gen_model.load_state_dict(torch.load(PATH,map_location='cpu'))
    if args.device != 'cpu':
        gen_model = gen_model.cuda()

    normalization = transforms.Normalize(mean=cifar10_mean, std=cifar10_std)

    gen_model.eval()

    top1 = AverageMeter()
    top5 = AverageMeter()
    test_iter = tqdm(test_loader, disable=args.local_rank not in [-1, 0])
    for step, (images, targets) in enumerate(test_iter):
        if args.device != 'cpu':
            images = images.cuda()
            targets = targets.cuda()

        # generator update
        recon,r = gen_model(images)
        recon_labels = app_model(normalization(recon))

        acc1, acc5 = accuracy(recon_labels, targets, (1, 5))
        top1.update(acc1[0], targets.shape[0])
        top5.update(acc5[0], targets.shape[0])
        test_iter.set_description(
            f"top1: {top1.avg:.2f}. top5: {top5.avg:.2f}. r: {r:.5f}. ")

    test_iter.close()
    # top1: 74.24. top5: 95.76. r: 0.0073.

class Simulator:
    def __init__(self,train=True,usemodel=True):
        self.opt = setup_opt()
        self.opt.resume = './checkpoint/cifar10-4K.5_best.pth.tar'
        if usemodel:
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
        dp = self.get_one_point((0,self.num_batches))
        # dp = self.get_multi_point([i*100 for i in range(1,8)])
        print (dp,self.num_batches)

if __name__ == '__main__':
    sim = Simulator(train=True)
    sim.test()
