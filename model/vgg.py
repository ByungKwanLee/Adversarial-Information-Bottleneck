import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

''' VGG Original Code by Pytorch '''
class VGG(nn.Module):

    def __init__(self, features, num_classes, mean, std):
        super(VGG, self).__init__()
        self.mean = mean
        self.std = std
        self.features = features
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = (x-self.mean) / self.std
        x = self.features(x)
        x = F.avg_pool2d(x, x.shape[2])
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

''' VGG Plain '''
class VGG_Plain_class(VGG):
    def __init__(self, features, num_classes, mean, std):
        super(VGG_Plain_class, self).__init__(features, num_classes, mean, std)

    def get_inference(self, x_adv):
        logit_adv = self(x_adv).detach()
        return logit_adv

''' VGG IFD '''
class VGG_IFD_class(VGG):
    def __init__(self, features, num_classes, mean, std):
        super(VGG_IFD_class, self).__init__(features, num_classes, mean, std)

    def forward(self, x, intermediate_propagate=0, pop=0):
        if intermediate_propagate == 0:
            x = (x-self.mean) / self.std
            for ind, l in enumerate(self.features):
                x = l(x)
                if pop==3 and ind==41:
                    return x

            x = F.avg_pool2d(x, x.shape[2])
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

        elif intermediate_propagate == 3:
            for ind, l in enumerate(self.features[41+1:]):
                x = l(x)
            x = F.avg_pool2d(x, x.shape[2])
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x


    def get_inference(self, x_adv):
        logit_adv = self(x_adv).detach()
        return logit_adv




''' VGG ReDefine '''
def make_layers(cfg, in_channels, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def VGG_Plain(in_channels, num_classes, mean, std, pretrained, batch_norm):
    model = VGG_Plain_class(make_layers(cfgs['vgg16'], in_channels=in_channels, batch_norm=batch_norm), num_classes, mean, std)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['vgg16' if not batch_norm else 'vgg16_bn'],
                                              progress=False)
        model.features.load_state_dict(state_dict, strict=False)
    return model


def VGG_IFD(in_channels, num_classes, mean, std, pretrained, batch_norm):
    model = VGG_IFD_class(make_layers(cfgs['vgg16'], in_channels=in_channels, batch_norm=batch_norm), num_classes, mean, std)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['vgg16' if not batch_norm else 'vgg16_bn'],
                                              progress=False)
        model.features.load_state_dict(state_dict, strict=False)
    return model






