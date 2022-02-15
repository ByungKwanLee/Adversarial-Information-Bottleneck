import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class InformativeFeaturePackage(nn.Module):
    def __init__(self, model, eps=0.03, attack_iter=10, IFD_iter=200, IFD_lr=0.1):
        super(InformativeFeaturePackage, self).__init__()
        self.model = model

        # PGD-based IFD attack hyper-parameter
        self.eps = eps
        self.attack_iter = attack_iter
        self.alpha = self.eps/attack_iter*2.3
        self.eta = 1e-2

        # IFD hyper-parameter
        self.IFD_iter = IFD_iter
        self.IFD_lr = IFD_lr
        self.cw_c = 0.1
        self.pgd_c = 10
        self.beta = 0.3
        self.grad = 1

        # define loss
        self.mse = nn.MSELoss(reduction='none')

        # softplus
        self.softplus = nn.Softplus()

    @staticmethod
    def grad_on_off(model, switch=False):
        for param in model.parameters():
            param.requires_grad=switch

    @staticmethod
    def kl_div(p, lambda_r):
        delta = 1e-10
        p_var = p.var(dim=[2, 3])
        q_var = (lambda_r.squeeze(-1).squeeze(-1)) ** 2

        eq1 = p_var / (q_var + delta)
        eq2 = torch.log((q_var + delta) / (p_var + delta))

        kld = 0.5 * (eq1 + eq2 - 1)

        return kld.mean()

    @staticmethod
    def sample_latent(latent_r, lambda_r):
        eps = torch.normal(0, 1, size=lambda_r.size()).cuda()
        return latent_r + lambda_r.mul(eps)


    def sample_robust_and_non_robust_latent(self, latent_r, lambda_r):

        var = lambda_r.square()
        r_var = latent_r.var(dim=(2,3)).view(-1)

        index = (var > r_var.max()).float()
        return index



    def find_features(self, input, labels, pop_number, forward_version=False):

        latent_r = self.model(input, pop=pop_number)
        lambda_r = torch.zeros([*latent_r.size()[:2],1,1]).cuda().requires_grad_()
        optimizer = torch.optim.Adam([lambda_r], lr=self.IFD_lr)

        for i in range(self.IFD_iter):

            lamb = self.softplus(lambda_r)
            latent_z = self.sample_latent(latent_r.detach(), lamb)
            outputs = self.model(latent_z.clone(), intermediate_propagate=pop_number)
            kl_loss = self.kl_div(latent_r.detach(), lamb)
            ce_loss = F.cross_entropy(outputs, labels)
            loss = ce_loss + self.beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        robust_lambda_r = lambda_r.clone().detach()

        # robust and non-robust index
        robust_index  = self.sample_robust_and_non_robust_latent(latent_r, self.softplus(robust_lambda_r))
        non_robust_index = 1-robust_index

        # robust and non-robust feature
        robust_latent_z     = latent_r * robust_index
        non_robust_latent_z = latent_r * non_robust_index

        robust_outputs = self.model(robust_latent_z.clone(), intermediate_propagate=pop_number).detach()
        _, robust_predicted = robust_outputs.max(1)

        non_robust_outputs = self.model(non_robust_latent_z.clone(), intermediate_propagate=pop_number).detach()
        _, non_robust_predicted = non_robust_outputs.max(1)

        if forward_version:
            return latent_r, robust_latent_z, non_robust_latent_z, \
                   robust_predicted, non_robust_predicted, robust_index, non_robust_index
        return latent_r, robust_latent_z, non_robust_latent_z, robust_predicted, non_robust_predicted

    @staticmethod
    def tanh_space(x):
        return 1 / 2 * (torch.tanh(x) + 1)

    @staticmethod
    def inverse_tanh_space(x):
        return 0.5 * torch.log((1 + x*2-1) / (1 - (x*2-1)))


    def NRF(self, images, labels):

        images = images.clone().detach().cuda()
        labels = labels.clone().detach().cuda()

        _, _, _, \
        _, _, _, non_robust_index \
            = self.find_features(images, labels, pop_number=3, forward_version=True)

        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        CE = nn.CrossEntropyLoss()
        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).cuda()
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        optimizer = torch.optim.Adam([w], lr=0.1)

        self.steps = 200
        for step in range(self.steps):
            # Get Adversarial Images
            adv_images = self.tanh_space(w)

            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            latent_r = self.model(adv_images, pop=3)
            outputs = self.model(latent_r.clone(), intermediate_propagate=3)
            f_loss = self.f(outputs, labels).sum()

            grad_latent = torch.autograd.grad(-f_loss, latent_r,
                                              retain_graph=True, create_graph=False)[0]

            cost_NR = torch.dist(latent_r, latent_r.detach() - non_robust_index * grad_latent.detach()) # something new method
            cost = L2_loss + self.cw_c * f_loss - self.grad*cost_NR

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update Adversarial Images
            _, pre = torch.max(outputs.detach(), 1)
            correct = (pre == labels).float()

            mask = (1 - correct) * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2

            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

            # Early Stop when loss does not converge.
            if step % (self.steps // 3) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()

        return best_adv_images

    def NRF2(self, images, labels):

        images = images.clone().detach().cuda()
        labels = labels.clone().detach().cuda()

        _, _, _, \
        _, _, robust_index, non_robust_index \
            = self.find_features(images, labels, pop_number=3, forward_version=True)

        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        CE = nn.CrossEntropyLoss()
        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).cuda()
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        optimizer = torch.optim.Adam([w], lr=0.1)

        self.steps = 200
        for step in range(self.steps):
            # Get Adversarial Images
            adv_images = self.tanh_space(w)

            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            latent_r = self.model(adv_images, pop=3)
            outputs = self.model(latent_r.clone(), intermediate_propagate=3)
            f_loss = self.f(outputs, labels).sum()

            grad_latent = torch.autograd.grad(-f_loss, latent_r,
                                              retain_graph=True, create_graph=False)[0]

            cost_NR = torch.dist(latent_r, latent_r.detach() - non_robust_index * grad_latent.detach()) # something new method
            cost = L2_loss + self.cw_c * f_loss + self.grad*cost_NR

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update Adversarial Images
            _, pre = torch.max(outputs.detach(), 1)
            correct = (pre == labels).float()

            mask = (1 - correct) * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2

            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

            # Early Stop when loss does not converge.
            if step % (self.steps // 3) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()

        return best_adv_images

    def RF(self, images, labels):

        images = images.clone().detach().cuda()
        labels = labels.clone().detach().cuda()

        _, _, _, \
        _, _, robust_index, non_robust_index \
            = self.find_features(images, labels, pop_number=3, forward_version=True)

        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        CE = nn.CrossEntropyLoss()
        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).cuda()
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        optimizer = torch.optim.Adam([w], lr=0.1)

        self.steps = 200
        for step in range(self.steps):
            # Get Adversarial Images
            adv_images = self.tanh_space(w)

            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            latent_r = self.model(adv_images, pop=3)
            outputs = self.model(latent_r.clone(), intermediate_propagate=3)
            f_loss = self.f(outputs, labels).sum()

            grad_latent = torch.autograd.grad(-f_loss, latent_r,
                                              retain_graph=True, create_graph=False)[0]

            cost_R = torch.dist(latent_r, latent_r.detach() - robust_index * grad_latent.detach()) # something new method
            cost = L2_loss + self.cw_c * f_loss - self.grad*cost_R

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update Adversarial Images
            _, pre = torch.max(outputs.detach(), 1)
            correct = (pre == labels).float()

            mask = (1 - correct) * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2

            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

            # Early Stop when loss does not converge.
            if step % (self.steps // 3) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()

        return best_adv_images

    def RF2(self, images, labels):

        images = images.clone().detach().cuda()
        labels = labels.clone().detach().cuda()

        _, _, _, \
        _, _, robust_index, non_robust_index \
            = self.find_features(images, labels, pop_number=3, forward_version=True)

        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        CE = nn.CrossEntropyLoss()
        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).cuda()
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        optimizer = torch.optim.Adam([w], lr=0.1)

        self.steps = 200
        for step in range(self.steps):
            # Get Adversarial Images
            adv_images = self.tanh_space(w)

            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            latent_r = self.model(adv_images, pop=3)
            outputs = self.model(latent_r.clone(), intermediate_propagate=3)
            f_loss = self.f(outputs, labels).sum()

            grad_latent = torch.autograd.grad(-f_loss, latent_r,
                                              retain_graph=True, create_graph=False)[0]

            cost_R = torch.dist(latent_r, latent_r.detach() - robust_index * grad_latent.detach()) # something new method
            cost = L2_loss + self.cw_c * f_loss + self.grad*cost_R

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update Adversarial Images
            _, pre = torch.max(outputs.detach(), 1)
            correct = (pre == labels).float()

            mask = (1 - correct) * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2

            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

            # Early Stop when loss does not converge.
            if step % (self.steps // 3) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()

        return best_adv_images
    # f-function in the paper
    @staticmethod
    def f(outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]))[labels].cuda()

        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())

        return torch.clamp(-1 * (i - j), min=0)


    def forward(self, input, labels, pop_number, noise=0):

        latent_r, robust_latent_z, non_robust_latent_z, \
        robust_predicted, non_robust_predicted, robust_index, non_robust_index \
            = self.find_features(input, labels, pop_number=pop_number, forward_version=True)

        robust_noise_output = self.model((robust_latent_z+robust_index*noise*torch.randn(robust_latent_z.shape).cuda()).clone(), intermediate_propagate=pop_number)
        non_robust_noise_output = self.model((non_robust_latent_z+(1-robust_index)*noise*torch.randn(non_robust_latent_z.shape).cuda()).clone(), intermediate_propagate=pop_number)

        _, robust_noise_predicted = robust_noise_output.max(1)
        _, non_robust_noise_predicted = non_robust_noise_output.max(1)

        return robust_predicted, non_robust_predicted, robust_noise_predicted, non_robust_noise_predicted




# sort_var, ele = var.squeeze().sort()
# import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt2

# ax1.plot(list(range(512)),list(sort_var), list(range(512)), list(r_var[ele]))
# ax2.plot(list(range(512)),list(sort_var), list(range(512)), list(r_var.max()*torch.ones(512).cuda()))
# ax3.plot(list(range(512)),list(sort_var), list(range(512)), list(torch.ones(512).cuda()))

# plt2.figure(1)
# plt2.plot(list(range(512)), list(sort_var), color='orange')
# plt2.plot(list(range(512)), list(r_var[ele]), 'b', linewidth=1)
# plt2.legend(['$\sigma$', '$\sigma_{s}$'])
# plt2.savefig("figure_1.png")
#
# plt.figure(2)
# plt.plot(list(range(512)), list(sort_var), color='orange')
# plt.plot(list(range(512)), list(r_var.max() * torch.ones(512).cuda()), 'b', linewidth=1)
# plt.legend(['$\sigma$','$M$'])
# plt.savefig("figure_2.png")
#
# plt.figure(3)
# plt.plot(list(range(512)), list(r_var[ele]), 'b', linewidth=1)
# plt.legend(['$\sigma_{s}$'])
# plt.savefig("figure_3.png")