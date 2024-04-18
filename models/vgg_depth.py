import torch
import torch.nn as nn
''' 
    这里选择和
        Factors of Transferability for a Generic ConvNet Representation
    内容保持同步
    
'''

cfg = {
    'A' : [64,     'M', 128,                'M', 256,      'M', 256, 256, 'M'],
    'B' : [64,     'M', 128, 128, 128,      'M', 256, 256, 'M', 256, 256, 'M'],
    'D' : [64, 64, 'M', 128, 128, 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
}

class VGG_Depth(nn.Module):

    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features

        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
    
        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, stride=1, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]
        
        layers += [nn.ReLU(inplace=True)]
        input_channel = l
    
    return nn.Sequential(*layers)

# def vgg11_bn():
#     return VGG(make_layers(cfg['A'], batch_norm=True))

# def vgg13_bn():
#     return VGG(make_layers(cfg['B'], batch_norm=True))

# def vgg16_bn():
#     return VGG(make_layers(cfg['D'], batch_norm=True))

# def vgg19_bn():
#     return VGG(make_layers(cfg['E'], batch_norm=True))

def deep8():
    return VGG(make_layers(cfg['A']))

def deep11():
    return VGG(make_layers(cfg['B']))

def deep13():
    return VGG(make_layers(cfg['C']))


