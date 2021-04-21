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

class TwoLayer(nn.Module):
    def __init__(self):
        super(TwoLayer, self).__init__()
        num_features = 128
        self.conv1 = nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features)
        self.conv3 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features)
        self.conv4 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features)
        self.conv5 = nn.Conv2d(num_features, 1, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = (F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = (F.relu(self.bn4(self.conv4(x))))
        x = self.pool(((self.conv5(x))))
        x = x.view(x.size(0), -1)
        x = F.tanh(x)
        x = x * 0.5 + 0.5
        return x

def disturb_main(param,datarange):
    sim_train = Simulator(train=True)
    return disturb_exp(sim_train.opt, sim_train.dataloader, sim_train.model, param, datarange)
    # sim_test = Simulator(train=False)
    # PATH = 'backup/sf.pth'
    # net = TwoLayer()
    # if sim_train.opt.device != 'cpu':
    #     net = net.cuda()
    # net.load_state_dict(torch.load(PATH,map_location='cpu'))
    # net.eval()

    # for i in range(10):
    #     s = time.perf_counter()
    #     print(net(torch.randn(1, 3, 32, 32)).shape)
    #     print(time.perf_counter()-s)
    # return 
    # for epoch in range(10):
    #     disturb_train(sim_train.opt, sim_train.dataloader, sim_train.model, net)
    #     disturb_test(sim_test.opt, sim_test.dataloader, sim_test.model, net)
    #     torch.save(net.state_dict(), PATH)

def check_accuracy(images,targets,model):
    normalization = transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    images = normalization(images)
    batch_size = targets.shape[0]
    outputs = model(images)
    entropy = nn.CrossEntropyLoss()
    loss = entropy(outputs, targets)
    acc1,_ = accuracy(outputs, targets, (1,5))
    return acc1,loss,torch.max(outputs,axis=1)[1]

def deepcod_main(param,datarange):
    from compression.deepcod import DeepCOD, orthorgonal_regularizer, init_weights
    sim_train = Simulator(train=True)
    sim_test = Simulator(train=False,usemodel=False)

    # data
    train_loader = sim_train.dataloader
    test_loader = sim_test.dataloader
    args = sim_train.opt

    # discriminator
    disc_model = sim_train.model

    # encoder+decoder
    PATH = 'backup/deepcod.pth'
    gen_model = DeepCOD()
    gen_model.apply(init_weights)
    # gen_model.load_state_dict(torch.load(PATH,map_location='cpu'))
    if args.device != 'cpu':
        gen_model = gen_model.cuda()
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    # optimizer = optim.SGD(gen_model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(gen_model.parameters(), lr=0.0001, betas=(0,0.9))
    normalization = transforms.Normalize(mean=cifar10_mean, std=cifar10_std)

    disc_model.eval()

    for epoch in range(1,1001):
        # training
        top1 = AverageMeter()
        top5 = AverageMeter()
        gen_model.train()
        train_iter = tqdm(train_loader, disable=args.local_rank not in [-1, 0])
        for step, (images, targets) in enumerate(train_iter):
            if args.device != 'cpu':
                images = images.cuda()
                targets = targets.cuda()

            recon = gen_model(images)
            # output of generated input
            recon_norm = normalization(recon)
            recon_labels,recon_features = disc_model(recon_norm,True)
            # output of original input
            images_norm = normalization(images)
            origin_labels,origin_features = disc_model(images_norm,True)

            reg_loss = orthorgonal_regularizer(gen_model.sample.weight,0.0001,args.device != 'cpu')
            # recon_loss = criterion_mse(images,recon)
            # loss += criterion_ce(recon_labels, targets)
            feat_loss = 0
            for origin_feat,recon_feat in zip(origin_features,recon_features):
                feat_loss += criterion_mse(origin_feat,recon_feat)
            loss = reg_loss + feat_loss
                    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc1, acc5 = accuracy(recon_labels, targets, (1, 5))
            top1.update(acc1[0], targets.shape[0])
            top5.update(acc5[0], targets.shape[0])
            train_iter.set_description(
                f"Train: {epoch:3}. "
                f"top1: {top1.avg:.2f}. top5: {top5.avg:.2f}. loss: {loss.cpu().item():.3f}. "
                f"reg_loss: {reg_loss.cpu().item():.3f}. feat_loss: {feat_loss.cpu().item():.3f}. ")
                # f"recon_loss: {recon_loss.cpu().item():.3f}. ")

        train_iter.close()

        # testing
        if epoch%5!=0:continue
        top1 = AverageMeter()
        top5 = AverageMeter()
        gen_model.eval()
        test_iter = tqdm(test_loader, disable=args.local_rank not in [-1, 0])
        with torch.no_grad():
            for step, (images, targets) in enumerate(test_iter):
                if args.device != 'cpu':
                    images = images.cuda()
                    targets = targets.cuda()

                recon = gen_model(images)
                # output of generated input
                recon_norm = normalization(recon)
                outputs,recon_features = disc_model(recon_norm,True)
                # output of original input
                images_norm = normalization(images)
                _,origin_features = disc_model(images_norm,True)

                reg_loss = orthorgonal_regularizer(gen_model.sample.weight,0.0001,args.device != 'cpu')
                # recon_loss = criterion_mse(images,recon)
                feat_loss = 0
                for origin_feat,recon_feat in zip(origin_features,recon_features):
                    feat_loss += criterion_mse(origin_feat,recon_feat)
                loss = reg_loss + feat_loss

                acc1, acc5 = accuracy(outputs, targets, (1, 5))
                top1.update(acc1[0], targets.shape[0])
                top5.update(acc5[0], targets.shape[0])
                test_iter.set_description(
                    f" Test: {epoch:3}. "
                    f"top1: {top1.avg:.2f}. top5: {top5.avg:.2f}. loss: {loss.cpu().item():.3f}. "
                    f"reg_loss: {reg_loss.cpu().item():.3f}. feat_loss: {feat_loss.cpu().item():.3f}. ")
                    # f"recon_loss: {recon_loss.cpu().item():.3f}. ")

        test_iter.close()
        torch.save(gen_model.state_dict(), PATH)


def disturb_exp(args, train_loader, model, param, datarange=None):
    from compression.transformer import Transformer
    # ROI_TF = Transformer(name='ROI')
    Qual_TF = Transformer(name='Quality')
    running_top1 = AverageMeter()
    Qual_TF.reset()
    model.eval()
    entropy = nn.CrossEntropyLoss()
    toMacroBlock = nn.AvgPool2d(kernel_size=(8,16), stride=(8,16), padding=0, ceil_mode=True)
    train_iter = tqdm(train_loader, disable=args.local_rank not in [-1, 0])
    for step, (images, targets) in enumerate(train_iter):
        if datarange is not None:
            if step<datarange[0]:continue
            elif step>=datarange[1]:break 
        cache_images = images.clone()
        normalization = transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
        images = normalization(images)
        batch_size = targets.shape[0]
        if args.device != 'cpu':
            images = images.cuda()
            targets = targets.cuda()

        # magics
        X = Variable(images,requires_grad=True) 
        # raw = ((X+1)/2.).data.cpu().numpy().transpose(0,2,3,1).clip(0,1)
        # fig = plot(raw)
        # plt.savefig(f'samples/real_{step:1}.png', bbox_inches='tight')
        # plt.close(fig)
        outputs = model(X)
        loss = entropy(outputs, targets)
        # loss = -outputs
        gradients = torch.autograd.grad(outputs=loss, inputs=X,
                                grad_outputs=torch.ones(loss.size()),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]
        impact = torch.norm(gradients.data, dim=1)
        impact = toMacroBlock(impact)
        impact = impact.view(impact.size(0),-1)
        # plot
        # samples = impact.view(impact.size(0),4,2).data.cpu().numpy()
        # fig = plot(samples)
        # plt.savefig(f'samples/max_class_{step:1}.png', bbox_inches='tight')
        # plt.close(fig)

        # loss map
        # C_params = torch.ones(impact.size())*0.4
        # loss_map = torch.zeros(impact.size())
        # for img_idx in range(64):
        #     print(img_idx)
        #     for pos in range(8):
        #         C_params[img_idx][pos] = 0.5
        #         Qual_TF.reset()
        #         tf_imgs = None
        #         for i,th_img in enumerate(cache_images):
        #             np_img = (th_img.permute(1,2,0).numpy()*255).astype(np.uint8)
        #             tf_img = Qual_TF.transform(image=np_img, C_param=C_params[i,:].numpy())
        #             tf_img = torch.from_numpy(tf_img/255).float().permute(2,0,1).unsqueeze(0)
        #             if tf_imgs is None:
        #                 tf_imgs = tf_img
        #             else:
        #                 tf_imgs = torch.cat((tf_imgs,tf_img),0)
        #         acc1,loss,pred = check_accuracy(tf_imgs,targets,model)
        #         cr = Qual_TF.get_compression_ratio()
        #         loss_map[img_idx,pos] = loss.cpu().item()
        #         C_params[img_idx][pos] = 0.4
        # loss_map /= torch.max(loss_map, dim=1, keepdim=True)[0]
        # samples = loss_map.view(loss_map.size(0),4,2).data.cpu().numpy()
        # fig = plot(samples)
        # plt.savefig(f'samples/loss_map_{step:1}.png', bbox_inches='tight')
        # plt.close(fig)


        # observ = torch.zeros((64,8))
        # C_params = torch.ones(impact.size())*0.4
        # for col in range(8):
        #     C_params[:,col] = 0.5
        #     Qual_TF.reset()
        #     tf_imgs = None
        #     for i,th_img in enumerate(cache_images):
        #         np_img = (th_img.permute(1,2,0).numpy()*255).astype(np.uint8)
        #         tf_img = Qual_TF.transform(image=np_img, C_param=C_params[i,:].numpy())
        #         tf_img = torch.from_numpy(tf_img/255).float().permute(2,0,1).unsqueeze(0)
        #         if tf_imgs is None:
        #             tf_imgs = tf_img
        #         else:
        #             tf_imgs = torch.cat((tf_imgs,tf_img),0)
        #     images = tf_imgs
        #     acc1,_,pred = check_accuracy(images,targets,model)
        #     cr = Qual_TF.get_compression_ratio()
        #     print('Metrics:',col,acc1,cr,targets.size())
        #     observ[:,col] = pred
        #     C_params[:,col] = 0.4
        # gt = targets.view(64,1).repeat(1,8)
        # maxval = torch.max(impact, dim=1, keepdim=True)[0]
        # test = maxval == impact
        # detect = observ == gt
        # for row in range(observ.size(0)):
        #     print(row,detect[row],impact[row])
        # print(torch.max(detect,dim=1))

        lower,upper,k = param+0.5
        k = int(k*8)
        k = max(k,0); k = min(k,7)
        # impact driven
        idx = torch.topk(impact,k,dim=1)[1]
        C_params = torch.ones(impact.size())*lower
        tf_imgs = None
        for i,th_img in enumerate(cache_images):
            np_img = (th_img.permute(1,2,0).numpy()*255).astype(np.uint8)
            C_params[i,idx[i]] = float(upper)
            tf_img = Qual_TF.transform(image=np_img, C_param=C_params[i,:].numpy())
            tf_img = torch.from_numpy(tf_img/255).float().permute(2,0,1).unsqueeze(0)
            if tf_imgs is None:
                tf_imgs = tf_img
            else:
                tf_imgs = torch.cat((tf_imgs,tf_img),0)
        images = tf_imgs
        acc1,_,_ = check_accuracy(images,targets,model)
        cr = Qual_TF.get_compression_ratio()
        # print('Impact:',acc1,cr,loss.cpu().item())
        running_top1.update(float(acc1))
        train_iter.set_description(
            f"Train Iter: {step+1:3}. Top1: {running_top1.avg:.3f} "
            f"Compression Rate: {cr:.3f}. ")

        # feature driven
        # C_param = np.array([0,0,0,-0.1,0,-.3])
        # ROI_TF.reset()
        # tf_imgs = None
        # for i,th_img in enumerate(cache_images):
        #     np_img = (th_img.permute(1,2,0).numpy()*255).astype(np.uint8)
        #     tf_img = ROI_TF.transform(image=np_img, C_param=C_param)
        #     tf_img = torch.from_numpy(tf_img/255).float().permute(2,0,1).unsqueeze(0)
        #     if tf_imgs is None:
        #         tf_imgs = tf_img
        #     else:
        #         tf_imgs = torch.cat((tf_imgs,tf_img),0)
        # images = tf_imgs
        # acc1,loss,_ = check_accuracy(images,targets,model)
        # cr = ROI_TF.get_compression_ratio()
        # print('Feature:',acc1,cr,loss.cpu().item())

        # jpeg
        # print('------------JPEG--------------')

        # C_params = torch.ones(impact.size())
        # for mul in range(1,11):
        #     Qual_TF.reset()
        #     tf_imgs = None
        #     for i,th_img in enumerate(cache_images):
        #         # print(i,C_params[i,:])
        #         # print(impact[i,:])
        #         np_img = (th_img.permute(1,2,0).numpy()*255).astype(np.uint8)
        #         tf_img = Qual_TF.transform(image=np_img, C_param=C_params[i,:].numpy()*mul/10.0)
        #         tf_img = torch.from_numpy(tf_img/255).float().permute(2,0,1).unsqueeze(0)
        #         if tf_imgs is None:
        #             tf_imgs = tf_img
        #         else:
        #             tf_imgs = torch.cat((tf_imgs,tf_img),0)
        #     images = tf_imgs
        #     acc1,loss,_ = check_accuracy(images,targets,model)
        #     cr = Qual_TF.get_compression_ratio()
        #     print('Metrics:',mul,acc1,cr,loss.cpu().item())

    train_iter.close()
    return running_top1.avg,Qual_TF.get_compression_ratio()

def disturb_train(args, train_loader, model, cnn_filter, datarange=None):
    model.eval()
    cnn_filter.train()
    running_loss = AverageMeter()
    entropy = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(cnn_filter.parameters(), lr=0.001, momentum=0.9)
    toMacroBlock = nn.AvgPool2d(kernel_size=8, stride=8, padding=0, ceil_mode=True)
    train_iter = tqdm(train_loader, disable=args.local_rank not in [-1, 0])
    for step, (images, targets) in enumerate(train_iter):
        if datarange is not None:
            if step<datarange[0]:continue
            elif step>=datarange[1]:break 
        normalization = transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
        images = normalization(images)
        # end transformation
        batch_size = targets.shape[0]
        if args.device != 'cpu':
            images = images.cuda()
            targets = targets.cuda()

        # magics
        X = Variable(images,requires_grad=True)
        # raw = ((X+1)/2.).data.cpu().numpy().transpose(0,2,3,1).clip(0,1)
        # fig = plot(raw)
        # plt.savefig(f'samples/real_{step:1}.png', bbox_inches='tight')
        # plt.close(fig)
        outputs = model(X)
        loss = entropy(outputs, targets)
        # loss = -outputs
        gradients = torch.autograd.grad(outputs=loss, inputs=X,
                                grad_outputs=torch.ones(loss.size()),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

        impact = torch.norm(gradients.data, dim=1)
        impact = toMacroBlock(impact)
        impact = impact.view(impact.size(0),-1)
        impact /= torch.max(impact, dim=1, keepdim=True)[0] 
        # samples = impact.view(impact.size(0),4,4).data.cpu().numpy()
        # fig = plot(samples)
        # plt.savefig(f'samples/max_class_{step:1}.png', bbox_inches='tight')
        # plt.close(fig)

        # zero gradient
        optimizer.zero_grad()

        # forward + backward + optimize
        pred = cnn_filter(images)
        loss = criterion(pred, impact)
        loss.backward()
        optimizer.step()

        running_loss.update(loss.cpu().item())

        train_iter.set_description(
            f"Train Iter: {step+1:3}. Loss: {loss.cpu().item():.3f} "
            f"Avg Loss: {running_loss.avg:.3f}. ")

    train_iter.close()

def disturb_test(args, train_loader, model, cnn_filter, datarange=None):
    model.eval()
    cnn_filter.eval()
    running_loss = AverageMeter()
    entropy = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    toMacroBlock = nn.AvgPool2d(kernel_size=8, stride=8, padding=0, ceil_mode=True)
    test_iter = tqdm(train_loader, disable=args.local_rank not in [-1, 0])
    for step, (images, targets) in enumerate(test_iter):
        if datarange is not None:
            if step<datarange[0]:continue
            elif step>=datarange[1]:break 
        normalization = transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
        images = normalization(images)
        # end transformation
        batch_size = targets.shape[0]
        if args.device != 'cpu':
            images = images.cuda()
            targets = targets.cuda()

        # magics
        X = Variable(images,requires_grad=True) 
        outputs = model(X)
        loss = entropy(outputs, targets)
        gradients = torch.autograd.grad(outputs=loss, inputs=X,
                                grad_outputs=torch.ones(loss.size()),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

        impact = torch.norm(gradients.data, dim=1)
        impact = toMacroBlock(impact)
        impact = impact.view(impact.size(0),-1)
        impact /= torch.max(impact, dim=1, keepdim=True)[0] 

        # forward + backward + optimize
        with torch.no_grad():
            pred = cnn_filter(images)
            loss = criterion(pred, impact)

        running_loss.update(loss.cpu().item())

        test_iter.set_description(
            f"Test Iter: {step+1:3}. Loss: {loss.cpu().item():.3f} "
            f"Avg Loss: {running_loss.avg:.3f}. ")

    test_iter.close()

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
