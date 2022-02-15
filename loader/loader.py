#!/usr/bin/env python

# torch package
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

# Custom package
from model.wideresnet import WideResNet_Plain, WideResNet_IFD
from model.vgg import VGG_Plain, VGG_IFD

# torchattacks toolbox
import torchattacks

def attack_loader(args, net):


    # Gradient Clamping based Attack
    if args.attack == "fgsm":
        return torchattacks.FGSM(model=net, eps=args.eps)

    elif args.attack == "bim":
        return torchattacks.BIM(model=net, eps=args.eps, alpha=1/255)

    elif args.attack == "pgd":
        return torchattacks.PGD(model=net, eps=args.eps,
                                alpha=args.eps/args.steps*2.3, steps=args.steps, random_start=True)

    elif args.attack == "cw":
        return torchattacks.CW(model=net, c=0.1, lr=0.1, steps=200)

    elif args.attack == "auto":
        return torchattacks.APGD(model=net, eps=args.eps)

    elif args.attack == "fab":
        return torchattacks.FAB(model=net, eps=args.eps, n_classes=args.n_classes)


    # Proposed attack
    elif args.attack == 'NRF':
        def f_attack(input, target):
            return net.NRF(input, target)
        return f_attack

    elif args.attack == 'NRF2':
        def f_attack(input, target):
            return net.NRF2(input, target)
        return f_attack

    elif args.attack == 'RF':
        def f_attack(input, target):
            return net.RF(input, target)
        return f_attack

    elif args.attack == 'RF2':
        def f_attack(input, target):
            return net.RF2(input, target)
        return f_attack




def network_loader(args, mean, std):
    print('Pretrained', args.pretrained)
    print('Batchnorm', args.batchnorm)
    if args.network == "wide":
        print('Wide Plain Network')
        return WideResNet_Plain(depth=28, in_channels=args.channel, num_classes=args.n_classes, widen_factor=10, dropRate=0.3, mean=mean, std=std)
    elif args.network == "vgg":
        print('VGG Plain Network')
        return VGG_Plain(in_channels=args.channel, num_classes=args.n_classes, mean=mean, std=std, pretrained=args.pretrained, batch_norm=args.batchnorm)


def IFD_network_loader(args, mean, std):
    print('Pretrained', args.pretrained)
    print('Batchnorm', args.batchnorm)
    if args.network == "wide":
        print('Wide IFD Network')
        return WideResNet_IFD(depth=28, in_channels=args.channel, num_classes=args.n_classes, widen_factor=10, dropRate=0.3, mean=mean, std=std)
    elif args.network == "vgg":
        print('VGG IFD Network')
        return VGG_IFD(in_channels=args.channel, num_classes=args.n_classes, mean=mean, std=std, pretrained=False, batch_norm=args.batchnorm)


def dataset_loader(args):

    args.mean=0.5
    args.std=0.25

    # Setting Dataset Required Parameters
    if args.dataset   == "svhn":
        args.n_classes = 10
        args.img_size  = 32
        args.channel   = 3
    elif args.dataset == "cifar10":
        args.n_classes = 10
        args.img_size  = 32
        args.channel   = 3
    elif args.dataset == "tiny":
        args.n_classes = 200
        args.img_size  = 64
        args.channel   = 3

    transform_train = transforms.Compose(
        [transforms.RandomCrop(args.img_size, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()]
    )

    transform_test = transforms.Compose(
        [transforms.ToTensor()]
    )

    # Full Trainloader/Testloader
    trainloader = torch.utils.data.DataLoader(dataset(args, True,  transform_train), batch_size=args.batch_size, shuffle=True, pin_memory=True)
    testloader  = torch.utils.data.DataLoader(dataset(args, False, transform_test),  batch_size=args.batch_size, shuffle=False, pin_memory=True)

    return trainloader, testloader


def dataset(args, train, transform):

        if args.dataset == "cifar10":
            return torchvision.datasets.CIFAR10(root=args.data_root, transform=transform, download=True, train=train)

        elif args.dataset == "svhn":
            return torchvision.datasets.SVHN(root=args.data_root, transform=transform, download=True,
                                    split="train" if train else "test")
        elif args.dataset == "tiny":
            return torchvision.datasets.ImageFolder(root=args.data_root+'/tiny-imagenet-200/train' if train \
                                    else args.data_root + '/tiny-imagenet-200/val', transform=transform)