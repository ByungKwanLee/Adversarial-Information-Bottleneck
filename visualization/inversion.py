from visualization.ops import tensor_to_img_array, torch_blur

import torch
from lucent.optvis import param, transform
import warnings
warnings.filterwarnings("ignore")


from visualization.ops import *
import torch.backends.cudnn as cudnn
from torchvision import transforms
from loader.loader import dataset_loader, IFD_network_loader, attack_loader

import scipy.ndimage as nd
import torch
import warnings

from lucent.optvis import render, param, transform, objectives
from lucent.modelzoo import inceptionv1
from lucent.misc.io.showing import animate_sequence
warnings.filterwarnings("ignore")

class SpInversion(object):
    def __init__(self, latent_z, model, dataset=None, fwd=False):
        super(SpInversion, self).__init__()
        #self.radn_img = torch.zeros((1, 3, 224,224)).to(device)
        self.model = model
        self.latent_z = latent_z
        self.cossim_pow = 2.0
        self.epochs = 512
        self.learning_rate = 5e-2
        self.fwd = fwd
        self.dataset = dataset

        if self.dataset == 'tiny':
            self.img_size = 64
            self.blur_constraint = -0.5
            self.cos_constant = 1e-6

        else:
            self.img_size = 32
            self.blur_constraint = 0.01
            self.cos_constant = 1e-10


    @staticmethod
    def grad_on_off(model, switch=False):
        for param in model.parameters():
            param.requires_grad=switch

    @staticmethod
    def normalize(x):
        mean = x.mean(dim=(2,3), keepdim=True)
        std = x.std(dim=(2,3), keepdim=True)
        return (x-mean) / (std + 1e-10)

    def invert(self, image, pop_number):
        r_transforms = [transform.pad(8, mode='constant', constant_value=.5),
                        transform.jitter(8),
                        transform.random_scale([0.9, 0.95, 1.05, 1.1] + [1] * 4),
                        transform.random_rotate(list(range(-5, 5)) + [0] * 5),
                        transform.jitter(2),
                        transform.Resize((self.img_size, self.img_size))]

        self.model.eval()

        if self.fwd:
            images = image.cuda()
            ref_acts = self.model(images, pop=pop_number).detach()
        else:
            ref_acts = self.latent_z.detach()

        params, image_f = param.image(self.img_size)
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)

        for i in range(self.epochs):
            acts = self.model(transform.compose(r_transforms)(image_f()), pop=pop_number)

            dot = (acts * ref_acts).sum()
            mag = torch.sqrt(torch.sum(ref_acts ** 2))
            cossim = dot / (self.cos_constant + mag)
            cos_loss = - dot * cossim ** self.cossim_pow

            with torch.no_grad():
                t_input_blurred = torch_blur(image_f())
            blur_loss = self.blur_constraint * torch.sum((image_f() - t_input_blurred) ** 2)

            tot_loss = cos_loss + blur_loss

            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step()

        return tensor_to_img_array(image_f())