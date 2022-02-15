import numpy as np
import torch
import torch.nn.functional as F
import os

from skimage.transform import resize
from PIL import Image, ImageDraw, ImageFont

def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def torch_blur(tensor, out_c=3, ):
    depth = tensor.shape[1]
    weight = np.zeros([depth, depth, out_c, out_c])
    for ch in range(depth):
        weight_ch = weight[ch, ch, :, :]
        weight_ch[ :  ,  :  ] = 0.5
        weight_ch[1:-1, 1:-1] = 1.0
    weight_t = torch.tensor(weight).float().cuda()
    conv_f = lambda t: F.conv2d(t, weight_t, None, 1, 1)

    return conv_f(tensor) / conv_f(torch.ones_like(tensor))

def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image


def lucid(ori_img, ori_inv, r_inv, nr_inv, label, adv=False, dataset=None):
    selectedFont = ImageFont.truetype(os.path.join('usr/share/fonts/', 'NanumGothic.ttf'), size=15)

    ori_img = Image.fromarray((resize(ori_img, (224, 224), anti_aliasing=True) * 255).astype(np.uint8))
    ori_inv = Image.fromarray((resize(ori_inv, (224, 224), anti_aliasing=True) * 255).astype(np.uint8))
    r_inv = Image.fromarray((resize(r_inv, (224, 224), anti_aliasing=True) * 255).astype(np.uint8))
    nr_inv = Image.fromarray((resize(nr_inv, (224, 224), anti_aliasing=True) * 255).astype(np.uint8))

    # save for paper
    ori_img.save("ori_img.png")
    ori_inv.save("ori_inv.png")
    r_inv.save("r_inv.png")
    nr_inv.save("nr_inv.png")
    # finish

    bg_img = Image.new("RGB", (224 * 4 + 20 * 5, 224 * 1 + 40), color=(255, 255, 255))

    bg_img.paste(ori_img, (20, 20))
    bg_img.paste(ori_inv, (224 * 1 + 20 * 2, 20))
    bg_img.paste(r_inv, (224 * 2 + 20 * 3, 20))
    bg_img.paste(nr_inv, (224 * 3 + 20 * 4, 20))

    if dataset == 'svhn':
        o_label = [str(i) for i in range(10)]
    elif dataset == 'cifar10':
        o_label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset == 'tiny':
        o_label = list(range(1, 201))

    draw = ImageDraw.Draw(bg_img)

    draw.text((20, 0), 'Image: ' + str(o_label[label[0]]), fill='blue', font=selectedFont)
    draw.text((224 * 1 + 20 * 2, 0), 'Intermediate: ' + str(o_label[label[1]]), fill='blue', font=selectedFont)
    draw.text((224 * 2 + 20 * 3, 0), 'Robust: ' + str(o_label[label[2]]), fill='blue', font=selectedFont)
    draw.text((224 * 3 + 20 * 4, 0), 'N-Robust: ' + str(o_label[label[3]]), fill='blue', font=selectedFont)

    return bg_img