#!/usr/bin/env python

# torch package
import os
import torch

# basic package
import sys
sys.path.append('.')
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings(action='ignore')


# custom package
from loader.argument_print import argument_testprint
from loader.loader import dataset_loader, IFD_network_loader, attack_loader
from model.IFP import InformativeFeaturePackage
# from visualization.ops import *

# cudnn enable
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# argument parser
parser = argparse.ArgumentParser(description='Joint Adversarial Defense')
parser.add_argument('--steps', default=10, type=int, help='adv. steps')
parser.add_argument('--eps', default=0.03, type=float, help='max norm')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--network', default='vgg', type=str, help='network name')
parser.add_argument('--data_root', default='./datasets', type=str, help='path to dataset')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
parser.add_argument('--save_dir', default='./experiment', type=str, help='save directory')
parser.add_argument('--batch_size', default=100, type=int, help='Batch size')
parser.add_argument('--pop_number', default=3, type=int, help='Batch size')
parser.add_argument('--datetime', default='00000000', type=str, help='checkpoint datetime')
parser.add_argument('--pretrained', default='false', type=str2bool, help='pretrained boolean')
parser.add_argument('--batchnorm', default='true', type=str2bool, help='batchnorm boolean')
parser.add_argument('--baseline', default='AT', type=str, help='baseline')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

checkpoint_name = args.baseline+'_'+args.network+'_'+args.dataset+'_'+args.datetime+'.pth'

# loading dataset, network, attack
_, testloader = dataset_loader(args)

# no remove module name
net = IFD_network_loader(args, mean=args.mean, std=args.std).cuda()
checkpoint_name = args.baseline+'_'+args.network+'_'+args.dataset+'_'+args.datetime+'.pth'
print('[IFD] ' + checkpoint_name +' has been Successfully Loaded')
state_dict = torch.load(os.path.join(args.save_dir, checkpoint_name))['model_state_dict']

# remove module name
# from collections import OrderedDict
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[7:] # remove `module.`
#     new_state_dict[name] = v
# state_dict = new_state_dict
#
# net.load_state_dict(state_dict)
# net.eval()


print('# of Model Parameters is: {:.3f}M\n'.format(sum(p.numel() for p in net.parameters() if p.requires_grad)/1e+6))
IFM = InformativeFeaturePackage(net, eps=args.eps, attack_iter=args.steps)

def distance(x_adv, x, attack):
    diff = (x_adv - x).view(x.size(0), -1)
    if attack in ('NRF', 'NRF2', 'cw', 'fab'):
        out = torch.sqrt((torch.sum(diff * diff, dim=1)/diff.size(1)).sum()/diff.size(0)).item()
        return out
    elif attack in ('fgsm', 'bim', 'pgd', 'auto'):
        out = torch.mean(torch.max(torch.abs(diff), 1)[0]).item()
        return out
    else:
        out = torch.sqrt((torch.sum(diff * diff, dim=1)/diff.size(1)).sum()/diff.size(0)).item()
        return out

def experiment_clean():

    # test arguement test print
    argument_testprint(args, checkpoint_name)

    correct = 0
    total = 0
    print('\n[IFD/Test] Under Testing ... Wait PLZ')
    for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):

        # dataloader parsing and generate adversarial examples
        inputs, targets = inputs.cuda(), targets.cuda()

        # Evaluation
        outputs = net.get_inference(inputs)

        # Test
        pred = torch.max(outputs, dim=1)[1]
        correct += torch.sum(pred.eq(targets)).item()
        total += targets.numel()

    print('[IFD/Test] Acc: {:.3f}'.format(100.*correct / total))


def experiment_robustness():

    for steps in [args.steps]:
        args.steps = steps
        attack_score = []

        # test arguemnet test print
        argument_testprint(args, checkpoint_name)

        # stack attack module
        attack_module = {}
        for attack_name in ['NRF', 'NRF2', 'RF', 'RF2']:
            args.attack = attack_name
            attack_module[attack_name]=attack_loader(args, IFM)
        for attack_name in ['fgsm', 'bim', 'pgd', 'cw', 'auto', 'fab']:
            args.attack = attack_name
            attack_module[attack_name]=attack_loader(args, net)

        # Measuring L2 distance
        l2_distance_list = []
        for key in attack_module:

            l2_distance = 0
            correct = 0
            total = 0
            print('\n[IFD/Test] Under Testing ... Wait PLZ')
            for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):

                # dataloader parsing and generate adversarial examples
                inputs, targets = inputs.cuda(), targets.cuda()
                adv_x = attack_module[key](inputs, targets) if args.eps != 0 else inputs
                l2_distance += distance(adv_x, inputs, key)

                # Evaluation
                outputs = net.get_inference(adv_x)

                # Test
                pred = torch.max(outputs, dim=1)[1]
                correct += torch.sum(pred.eq(targets)).item()
                total += targets.numel()

            print('[IFD/{}] Acc: {:.3f} ({:.3f})'.format(key, 100.*correct / total, l2_distance/(batch_idx+1)))
            attack_score.append(100.*correct / total)
            l2_distance_list.append(l2_distance/(batch_idx+1))


        print('\n----------------Summary----------------')
        print(steps, ' steps attack')
        for key, l2, score in zip(attack_module, l2_distance_list, attack_score):
            print(str(key),' : ', score, ' ({:.4f})'.format(l2))
        print('---------------------------------------\n')



if __name__ == '__main__':
    experiment_clean()
    experiment_robustness()
