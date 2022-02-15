#!/usr/bin/env python

# torch package
import torch

# basic package
import os
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image

# custom package
from loader.argument_print import argument_print
from loader.loader import dataset_loader, IFD_network_loader, attack_loader
from model.IFP import InformativeFeaturePackage


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
parser = argparse.ArgumentParser(description='Informative Feature Decomposition')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--steps', default=10, type=int, help='adv. steps')
parser.add_argument('--eps', default=0.03, type=float, help='max norm')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--network', default='vgg', type=str, help='network name')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
parser.add_argument('--data_root', default='./datasets', type=str, help='path to dataset')
parser.add_argument('--epoch', default=0, type=int, help='epoch number')
parser.add_argument('--attack', default='pgd', type=str, help='attack type')
parser.add_argument('--save_dir', default='./experiment', type=str, help='save directory')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--pop_number', default=3, type=int, help='Batch size')
parser.add_argument('--prior', default='AT', type=str, help='Plain or AT')
parser.add_argument('--prior_datetime', default='00000000', type=str, help='checkpoint datetime')
parser.add_argument('--pretrained', default='false', type=str2bool, help='pretrained boolean')
parser.add_argument('--batchnorm', default='true', type=str2bool, help='batchnorm boolean')
parser.add_argument('--vis_atk', default='True', type=str2bool, help='is attacked image?')
args = parser.parse_args()

# 05071522
# 05140308
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
from visualization.inversion import SpInversion
from visualization.ops import lucid, check_dir

# loading dataset, network, and attack
trainloader, testloader = dataset_loader(args)

# no remove module name
net = IFD_network_loader(args, mean=args.mean, std=args.std).cuda()
checkpoint_name = args.prior+'_'+args.network+'_'+args.dataset+'_'+args.prior_datetime+'.pth'
print('[IFD] ' + checkpoint_name +' has been Successfully Loaded')
state_dict = torch.load(os.path.join(args.save_dir, checkpoint_name))['model_state_dict']

# remove module name
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
state_dict = new_state_dict

net.load_state_dict(state_dict)
net.eval()

for param in net.parameters():
    param.requires_grad = False

IFM = InformativeFeaturePackage(net, eps=args.eps, attack_iter=args.steps)

# loading attack method
attack = attack_loader(args, net)

# argument print
argument_print(args, checkpoint_name)


def visualization():

    net.eval()
    save_dir = './results_' + str(args.dataset) + '_' + str(args.prior) + '_' + str(args.network) + '_' + str(
        args.pop_number)
    check_dir(save_dir)

    for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):

        # dataloader parsing and generate adversarial examples
        inputs, targets = inputs.cuda(), targets.cuda()
        latent_r, robust_latent_z, non_robust_latent_z, robust_predicted, non_robust_predicted, _, _ = IFM.find_features(inputs, targets, args.pop_number, forward_version=True)

        if args.vis_atk:
            adv_x = attack(inputs, targets) if args.eps != 0 else inputs
            adv_latent_r, adv_robust_latent, adv_non_robust_latent, adv_robust_predicted, adv_non_robust_predicted, _, _ = IFM.find_features(adv_x, targets, args.pop_number, forward_version=True)
            _, adv_pred = net(adv_latent_r.clone(), intermediate_propagate=args.pop_number).max(1)

            adv_label = [targets.item(), adv_pred.item(), adv_robust_predicted.item(), adv_non_robust_predicted.item()]
            adv_img = np.transpose(adv_x.squeeze().cpu().detach().numpy(), [1, 2, 0])
            adv_inv = SpInversion(adv_latent_r.clone(), net, dataset=args.dataset).invert(inputs, args.pop_number).squeeze()
            adv_r_inv = SpInversion(adv_robust_latent.clone(), net, dataset=args.dataset).invert(inputs, args.pop_number).squeeze()
            adv_nr_inv = SpInversion(adv_non_robust_latent.clone(), net, dataset=args.dataset).invert(inputs, args.pop_number).squeeze()
            adv_img = lucid(adv_img, adv_inv, adv_r_inv, adv_nr_inv, adv_label, dataset=args.dataset)

            adv_img.save(save_dir + '/adv_img%d.png' % (batch_idx))
            print("\n [*] ADV Img%d is saved" % (batch_idx))

        else:
            _, t_pred = net(inputs).max(1)
            _, pred = net(latent_r.clone(), intermediate_propagate=args.pop_number).max(1)

            for param in net.parameters():
                param.requires_grad = False

            clean_label = [targets.item(), t_pred.item(), robust_predicted.item(), non_robust_predicted.item()]
            ori_img = np.transpose(inputs.squeeze().cpu().detach().numpy(), [1, 2, 0])
            inv = SpInversion(latent_r.clone(), net, dataset=args.dataset).invert(inputs, args.pop_number).squeeze()
            r_inv = SpInversion(robust_latent_z.clone(), net, dataset=args.dataset).invert(inputs, args.pop_number).squeeze()
            nr_inv = SpInversion(non_robust_latent_z.clone(), net, dataset=args.dataset).invert(inputs, args.pop_number).squeeze()
            clean_img = lucid(ori_img, inv, r_inv, nr_inv, clean_label, dataset=args.dataset)

            clean_img.save(save_dir + '/img%d.png' % (batch_idx))
            print("\n [*] CLEAN Img%d is saved" % (batch_idx))

if __name__ == "__main__":
    visualization()
