import argparse
import random
import numpy as np
import os
import sys
import time
import shutil
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from models import *
from diffusion import Model


parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-ct', '--cifar-type', default='10', type=int, metavar='CT', help='10 for cifar10,100 for cifar100 (default: 10)')

best_prec = 0
alpha = 1

def main():
    global args, best_prec
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    # Model building
    print('=> Building model...')
    if use_gpu:

        model = resnet44_cifar(num_classes=10)

        # mkdir a new folder to store the checkpoint and best model
        if not os.path.exists('result'):
            os.makedirs('result')
        fdir = 'result/resnet44_cifar10'
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        # adjust the lr according to the model type
        if isinstance(model, (ResNet_Cifar, PreAct_ResNet_Cifar)):
            model_type = 1
        else:
            print('model type unrecognized...')
            return

        model = model.cuda()
        adver = Model(128, 3, [1, 2, 2, 2], 2, [16,], 0.1, 3, 32).cuda()
        adver.load_state_dict(torch.load('/home/liuluping/llp/Workspace/fast_ddim/output4/pre_train/diffusion_models_converted/ema_diffusion_cifar10_model/model-790000.ckpt'))
        criterion = nn.CrossEntropyLoss().cuda()
        optim_m = optim.Adam(model.parameters(), args.lr)
        optim_a = optim.Adam(adver.parameters(), args.lr)
        cudnn.benchmark = True
    else:
        print('Cuda is not available!')
        return

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint "{}"'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            model.load_state_dict(checkpoint['state_dict'])
            optim_m.load_state_dict(checkpoint['optim_m'])
            optim_a.load_state_dict(checkpoint['optim_a'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading and preprocessing
    # CIFAR10
    if args.cifar_type == 10:
        print('=> loading cifar10 data...')
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.5, 0.5, 0.5])

        train_dataset = torchvision.datasets.CIFAR10(
            root='~/llp/Datasets/cifar10',
            train=True, 
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        test_dataset = torchvision.datasets.CIFAR10(
            root='~/llp/Datasets/cifar10',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    if args.evaluate:
        validate(testloader, model, adver, criterion)
        return

    flag = random.randint(10000, 99999)
    writer = SummaryWriter(f'board/{flag}')
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch

        train(trainloader, model, adver, criterion, optim_m, epoch, writer)

        if epoch >= 3:
            noise(trainloader, model, adver, criterion, optim_a, epoch, writer)

        # evaluate on test set
        # prec = validate(testloader, model, adver, criterion, len(trainloader), epoch, writer)

        # remember best precision and save checkpoint
        # is_best = prec > best_prec
        # best_prec = max(prec,best_prec)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optim_m': optim_m.state_dict(),
            'optim_a': optim_a.state_dict()
        }, None, fdir)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(trainloader, model, adver, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    adver.eval()
    global alpha

    end = time.time()
    for i, (input, target) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.cuda(), target.cuda()

        # compute output
        input = torch.clip(input, -1, 1).detach()
        # noise, input_ = adver(input), None
        # alpha_ = torch.rand(1).item() * alpha
        # if epoch >= 5:
        #     input_ = torch.clip(input + alpha_ * noise, -1, 1)
        # else:
        #     input_ = input
        output = model(input.detach())
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec = accuracy(output, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            writer.add_scalar('train/loss', loss.item(), global_step=epoch*len(trainloader)+i)
            writer.add_scalar('train/prec', prec.item(), global_step=epoch * len(trainloader) + i)
            # writer.add_scalar('train/alpha', alpha, global_step=epoch * len(trainloader) + i)
            # if prec.item() > 90 and alpha_ > alpha / 2:
            #     alpha += 0.0005
            if i % 100 == 0:
                writer.add_images('train/clean', (input[:24]+1)/2, global_step=epoch * len(trainloader) + i)
                if prec.item() > 90:
                    torch.save(model, f'/home/liuluping/llp/Workspace/adversarial/model-{prec.item()}-{epoch * len(trainloader) + i}.pth')
                # writer.add_images('train/noise', (alpha_ * noise[:24]+1)/2, global_step=epoch * len(trainloader) + i)
                # writer.add_images('train/input', (input_[:24]+1)/2, global_step=epoch * len(trainloader) + i)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   epoch, i, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))


def noise(trainloader, model, adver, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    adver.eval()

    end = time.time()
    for i, (input, target) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.cuda(), target.cuda()
        shape = input.shape
        # compute output
        input = torch.clip(input, -1, 1)
        input = input.detach().requires_grad_(True)
        t = (torch.ones(shape[0]) * 50).to(input.device)
        if input.grad is not None:
            input.grad.data.fill_(0)
        noise = adver(input, t)
        clean = input + denoising_precessing(input, noise, t, input.device)
        output = model(clean)
        loss = -criterion(output, target)

        # measure accuracy and record loss
        prec = accuracy(output, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()
        grad = input.grad
        shape = grad.shape
        # grad = (grad - torch.mean(grad)) / torch.std(grad)
        grad = torch.sign(grad) * 0.1
        with torch.no_grad():
            grad = torch.clip(grad, -1, 1)
            input_ = torch.clip(input + grad, -1, 1)
            output = model(input_)
            loss_1 = criterion(output, target)
            prec_1 = accuracy(output, target)[0]

            noise = adver(input_, t)
            loss_n = torch.norm(grad-noise, p=2) * 0.05
            noise = torch.clip(noise, -1, 1)
            clean_ = input_ + denoising_precessing(input_, noise, t, input_.device)

            output = model(clean_)
            loss_2 = criterion(output, target)
            prec_2 = accuracy(output, target)[0]
        # optimizer.zero_grad()
        # loss_n.backward()
        # optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            writer.add_scalar('noise/loss_n', loss_n.item(), global_step=epoch * len(trainloader) + i)
            writer.add_scalar('noise/loss_1', loss_1.item(), global_step=epoch*len(trainloader)+i)
            writer.add_scalar('noise/prec_1', prec_1.item(), global_step=epoch * len(trainloader) + i)
            writer.add_scalar('noise/loss_2', loss_2.item(), global_step=epoch * len(trainloader) + i)
            writer.add_scalar('noise/prec_2', prec_2.item(), global_step=epoch * len(trainloader) + i)

            if i % 100 == 0:
                writer.add_images('noise/clean', (input[:24]+1)/2, global_step=epoch * len(trainloader) + i)
                writer.add_images('noise/dirty', (input_[:24] + 1) / 2, global_step=epoch * len(trainloader) + i)
                writer.add_images('noise/denoise', (clean_[:24] + 1) / 2, global_step=epoch * len(trainloader) + i)
                writer.add_images('noise/noise', (noise[:24]+1)/4, global_step=epoch * len(trainloader) + i)
                writer.add_images('noise/grad', (grad[:24] + 1) / 2, global_step=epoch * len(trainloader) + i)
            print(epoch, i, loss.item(), loss_n.item(), prec.item())
            # print('Epoch: [{0}][{1}/{2}]\t'
            #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #       'Loss_n {loss.val:.4f} ({loss.avg:.4f})\t'
            #       'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
            #        epoch, i, len(trainloader), batch_time=batch_time,
            #        data_time=data_time, loss=losses, top1=top1))

def denoising_precessing(x, e, t, device):
    betas = np.linspace(
        0.0001, 0.02, 1000, dtype=np.float64
    )
    b = torch.from_numpy(betas).float().to(device)
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    at = compute_alpha(b, t.long())
    at_next = compute_alpha(b, (t*0+1).long())
    denoising = (at_next - at) * ((1 / (at.sqrt() * (at.sqrt() + at_next.sqrt()))) * x - \
                                1 / (at.sqrt() * (((1 - at_next) * at).sqrt() + ((1 - at) * at_next).sqrt())) * e)

    return denoising

def validate(val_loader, model, adver, criterion, len_epoch, epoch, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    adver.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.cuda(), target.cuda()

            # compute output
            input = torch.clip(input, -1, 1)
            noise = adver(input)
            alpha_ = torch.rand(1).item() * alpha
            input_ = torch.clip(input + alpha_ * noise, -1, 1)
            output = model(input_)
            loss_1, loss_2 = criterion(output, target), torch.norm(noise, p=2)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                writer.add_scalar('valid/loss_1', loss_1.item(), global_step=epoch * len_epoch + i)
                writer.add_scalar('valid/loss_2', loss_2.item(), global_step=epoch * len_epoch + i)
                writer.add_scalar('valid/prec', prec.item(), global_step=epoch * len_epoch + i)
                writer.add_images('valid/clean', (input[:24] + 1)/2, global_step=epoch * len_epoch + i)
                writer.add_images('valid/noise', (alpha_ * noise[:24] + 1)/2, global_step=epoch * len_epoch + i)
                writer.add_images('valid/input', (input_[:24] + 1)/2, global_step=epoch * len_epoch + i)
                # if i == 0:
                #     print(noise)
                print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))

    print(' * Prec {top1.avg:.3f}% '.format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, fdir):
    filepath = os.path.join(fdir, 'checkpoint.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, 'model_best.pth.tar'))



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__=='__main__':
    main()

