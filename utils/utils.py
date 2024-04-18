import sys
import logging
import os
import numpy as np
from tqdm import tqdm
import math
import collections
from abc import abstractmethod

# import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models

import scipy
from scipy.ndimage import grey_dilation, grey_erosion

import global_settings as settings

from dataset.prototype import Prototype
from dataset.segmentation.voc import VOCDataset
from dataset.p3m import p3m

def get_network(arch, class_num):
    """ return given network
    """
    if arch == 'vgg16':
        from models.vgg import vgg16
        net = vgg16(class_num)
    elif arch == 'vgg13':
        from models.vgg import vgg13
        net = vgg13(class_num)
    elif arch == 'vgg11':
        from models.vgg import vgg11
        net = vgg11(class_num)
    elif arch == 'vgg11_small':
        from models.vgg import vgg11_small
        net = vgg11_small(class_num)
    elif arch == 'vgg11_tiny':
        from models.vgg import vgg11_tiny
        net = vgg11_tiny(class_num)
    elif arch == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn(class_num)
    elif arch == 'densenet121':
        from models.densenet import densenet121
        net = densenet121(class_num)
    elif arch == 'densenet161':
        from models.densenet import densenet161
        net = densenet161(class_num)
    elif arch == 'densenet169':
        from models.densenet import densenet169
        net = densenet169(class_num)
    elif arch == 'densenet201':
        from models.densenet import densenet201
        net = densenet201(class_num)
    elif arch == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet(class_num)
    elif arch == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif arch == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif arch == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif arch == 'xception':
        from models.xception import xception
        net = xception()
    elif arch == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif arch == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif arch == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif arch == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif arch == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif arch == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif arch == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif arch == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif arch == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif arch == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif arch == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif arch == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif arch == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif arch == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif arch == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif arch == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif arch == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif arch == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif arch == 'mobilenet_v3_small':
        from models.mobilenetv3 import mobilenetv3
        net = mobilenetv3(mode='small')
    elif arch == 'mobilenet_v3_large':
        from models.mobilenetv3 import mobilenetv3
        net = mobilenetv3(mode='large')
    elif arch == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif arch == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif arch == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif arch == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif arch == 'seresnet34':
        from models.senet import seresnet34 
        net = seresnet34()
    elif arch == 'seresnet50':
        from models.senet import seresnet50 
        net = seresnet50()
    elif arch == 'seresnet101':
        from models.senet import seresnet101 
        net = seresnet101()
    elif arch == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    return net

def get_torchvision_network(arch, class_num=1000, pretrain=False):
    model = ''
    weights = ''
    if pretrain:
        if arch == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1
            model = models.resnet34(weights=weights, num_classes=class_num)
        elif arch == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights, num_classes=class_num)
        elif arch == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            model = models.resnet50(weights=weights, num_classes=class_num)
        elif arch == 'densenet169':
            weights = models.DenseNet161_Weights.IMAGENET1K_V1
            model = models.densenet161(weights=weights, num_classes=class_num)
        elif arch == 'mobilenet_v3_small':
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            model = models.mobilenet_v3_small(weights=weights, num_classes=class_num)
        elif arch == 'vgg16':
            weights = models.VGG16_BN_Weights.IMAGENET1K_V1
            model = models.vgg16_bn(weights=weights, num_classes=class_num)
        elif arch == 'inception_v3':
            weights = models.Inception_V3_Weights.IMAGENET1K_V1
            model = models.inception_v3(weights=weights, num_classes=class_num)
        elif arch == 'mobilenet_v2':
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            model = models.mobilenet_v2(weights=weights, num_classes=class_num)
        elif arch == 'alexnet':
            weights = models.AlexNet_Weights.IMAGENET1K_V1
            model = models.alexnet(weights=weights, num_classes=class_num)
        elif arch == 'swin_s':
            weights = models.Swin_S_Weights.IMAGENET1K_V1
            model = models.swin_s(weights=weights, num_classes=class_num)
        elif arch == 'vit_b_16':
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1
            model = models.vit_b_16(weights=weights, num_classes=class_num)
        else:
            raise RuntimeError(f"Not model available for {arch} on imagenet")
    else:
        if arch == 'resnet34':
            model = models.resnet34(weights=None, num_classes=class_num)
        elif arch == 'resnet18':
            model = models.resnet18(weights=None, num_classes=class_num)
        elif arch == 'resnet50':
            model = models.resnet50(weights=None, num_classes=class_num)
        elif arch == 'densenet161':
            model = models.densenet161(weights=None, num_classes=class_num)
        elif arch == 'mobilenet_v3_small':
            model = models.mobilenet_v3_small(weights=None, num_classes=class_num)
        elif arch == 'vgg16':
            model = models.vgg16_bn(weights=None, num_classes=class_num)
        elif arch == 'vgg11':
            model = models.vgg11(weights=None, num_classes=class_num)
        elif arch == 'inception_v3':
            model = models.inception_v3(weights=None, num_classes=class_num)
        elif arch == 'inception_v3':
            model = models.inception_v3(weights=None, num_classes=class_num)
        elif arch == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=None, num_classes=class_num)
        elif arch == 'alexnet':
            model = models.alexnet(weights=None, num_classes=class_num)
        elif arch == 'swin_s':
            model = models.swin_s(weights=None, num_classes=class_num)
        elif arch == 'vit_b_16':
            model = models.vit_b_16(weights=None, num_classes=class_num)
        else:
            raise RuntimeError(f"Not model available for {arch} from torchvision")

    return model

def load_weight(model, weight_path, device):
    pre_weights = torch.load(weight_path, map_location=device)
    state_dict = {k:v for k, v in pre_weights.items()}
    model.load_state_dict(state_dict, strict=True)

def load_weight_till_target_layer(model, weight_path, target_layer):
    pre_weights = torch.load(weight_path)
    state_dict = {k:v for k, v in pre_weights.items()}
    layer_names = list(pre_weights.keys())
    target_layer_name_dict = {
        3 : 'layer1.0.bn2.num_batches_tracked',
        11 : 'layer3.0.bn2.num_batches_tracked'
    }
    
    assert target_layer in target_layer_name_dict.keys()
    target_layer_name = target_layer_name_dict[target_layer]
    
    for index, name in enumerate(layer_names):
        if target_layer_name == name:
            layer_names = layer_names[:index+1]
            break
    
    state_dict = {k:v for k, v in pre_weights.items() if k in layer_names}
    model.load_state_dict(state_dict, strict=False)
    
    
def replace_new_classifier(model, arch, class_num, device):
    if arch == 'resnet18' or arch == 'resnet50':
        feature_dim = model.fc.in_features
        model.fc = nn.Linear(feature_dim, class_num).to(device)
    elif arch == 'mobilenet_v2':
        fc = getattr(model, 'classifier')
        feature_dim = fc[1].in_features
        ln = nn.Linear(feature_dim, class_num).to(device)
        fc[1] = ln
    elif arch == 'alexnet':
        in_feature, out_feature = model.classifier[4].in_features, model.classifier[4].out_features
        model.classifier[4] = nn.Linear(in_feature, out_feature).to(device)
        in_feature = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_feature, class_num).to(device)
    # inception不包含辅助分类器
    elif arch == 'inception_v3':
        feature_dim = model.fc.in_features
        model.fc = nn.Linear(feature_dim, class_num).to(device)
    elif arch == 'swin_s':
        feature_dim = model.head.in_features
        model.head = nn.Linear(feature_dim, class_num).to(device)
    elif arch == 'vit_b_16':
        feature_dim = model.heads.head.in_features
        model.heads.head = nn.Linear(feature_dim, class_num).to(device)
    elif arch == 'densenet169':
        feature_dim = model.classifier.in_features
        model.classifier = nn.Linear(feature_dim, class_num).to(device)
    elif arch == 'vgg11' or arch == 'vgg13' or arch == 'vgg16' or arch == 'vgg11_small' or arch == 'vgg11_tiny':
        in_feature, out_feature = model.classifier[3].in_features, model.classifier[3].out_features
        model.classifier[3] = nn.Linear(in_feature, out_feature).to(device)
        in_feature = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_feature, class_num).to(device)
    else:
        print('Replace: Model not support!')

def get_training_transform(dataset, resolution):
    transform = None
    target_transform = None
    if dataset == 'nyu':
        transform = transforms.Compose([
            transforms.Resize(256, 256),  
            transforms.ToTensor(),
            transforms.Normalize(settings.mean_std['imagenet']['mean'], settings.mean_std['imagenet']['std'])
        ])
        target_transform = transforms.Compose([
            transforms.Resize(),  
            transforms.ToTensor(),
        ])
    elif dataset == 'voc':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(settings.mean_std['imagenet']['mean'], settings.mean_std['imagenet']['std'])
        ])
        target_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(settings.mean_std['imagenet']['mean'], settings.mean_std['imagenet']['std'])
        ])
        # from dataset.segmentation.presets import SegmentationPresetTrain
        # transform = SegmentationPresetTrain(base_size=520, crop_size=480, backend='pil', use_v2=True)
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop([resolution, resolution]),
            transforms.ToTensor(),
            transforms.Normalize(settings.mean_std['imagenet']['mean'], settings.mean_std['imagenet']['std'])
        ])
    return transform, target_transform

def get_test_transform(dataset, resolution):
    transform = None
    target_transform = None
    if dataset == 'nyu':
        transform = transforms.Compose([
            transforms.Resize(256, 256),  
            transforms.ToTensor(),
            transforms.Normalize(settings.mean_std['imagenet']['mean'], settings.mean_std['imagenet']['std'])
        ])
        target_transform = transforms.Compose([
            transforms.Resize(),  
            transforms.ToTensor(),
        ])
    elif dataset == 'voc':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(settings.mean_std['imagenet']['mean'], settings.mean_std['imagenet']['std'])
        ])
        target_transform = transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(settings.mean_std['imagenet']['mean'], settings.mean_std['imagenet']['std'])
        ])
        # from dataset.segmentation.presets import SegmentationPresetEval
        # tranform = SegmentationPresetEval(base_size=520, backend='pil', use_v2=True)
    else:
        transform = transforms.Compose([
            transforms.Resize(256 if resolution < 256 else resolution),
            transforms.CenterCrop([resolution, resolution]),
            transforms.ToTensor(),
            transforms.Normalize(settings.mean_std['imagenet']['mean'], settings.mean_std['imagenet']['std'])
        ])
    return transform, target_transform

# 返回dataloader(train and test)
def get_dataset_and_loader(dataset, batch_size, resolution=None, class_num=None, train_sample_num=None, test_sample_num=None, total_num=None, ratio=None):
    train_loader, test_loader = '', ''
    
    # 若为同时对image和target作用的transforms，将会保存在tranform_train(test)即第一个返回值中
    transform_train, target_transform_train = get_training_transform(dataset, resolution)
    transform_test, target_transform_test = get_test_transform(dataset, resolution)
    
    classification_datasets = ['cifar100', 'food101', 'imagenet', 'flowers102', 'cifar10', 'cub200', 'caltech256', 'dogs', 'sun397']
    domainnet_datasets = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    multi_task_datasets = ['nyu', 'voc', 'p3m']

    if dataset in classification_datasets:
        if dataset == 'cifar10' and train_sample_num is None and total_num is not None:
            train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            test_ds = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        else:
            train_ds = Prototype(root=f'./data/{dataset}', transform=transform_train, split='train', class_num=class_num, sample_num=train_sample_num, total_num=total_num,  ratio=ratio)
            test_ds = Prototype(root=f'./data/{dataset}', transform=transform_test, split='test', class_num=class_num, sample_num=test_sample_num, total_num=total_num, ratio=ratio)
    elif dataset in domainnet_datasets:
        train_ds = Prototype(root=f'./data/DomainNet/{dataset}', transform=transform_train, split='train', class_num=class_num, sample_num=train_sample_num, total_num=total_num,  ratio=ratio)
        test_ds = Prototype(root=f'./data/DomainNet/{dataset}', transform=transform_test, split='test', class_num=class_num, sample_num=test_sample_num, total_num=total_num, ratio=ratio)
    elif dataset in multi_task_datasets:
        if dataset == 'nyu':
            pass
            # train_ds = NYUDataset(split='train', rgb_transform=transform_train, depth_transform=target_transform_train)
            # test_ds = NYUDataset(split='test',rgb_transform=transform_test, depth_transform=target_transform_test)
        elif dataset == 'voc':
            # train_ds = torchvision.datasets.VOCSegmentation(root='/datasets/VOCdevkit/VOC2012', year='2012', image_set='train', transforms=transform_train)
            # test_ds = torchvision.datasets.VOCSegmentation(root='/datasets/VOCdevkit/VOC2012', year='2012', image_set='val', transforms=transform_test)
            train_ds = VOCDataset(split='train', transform=transform_train, target_transform=target_transform_train)
            test_ds = VOCDataset(split='test', transform=transform_test, target_transform=target_transform_test)
        elif dataset == 'p3m':
            train_ds = p3m.MattingDataset(split='train', transform=p3m.MattingTransform())
            test_ds = p3m.MattingDataset(split='test', transform=p3m.MattingTransform())
            # train_ds = p3m.MattingDataset(split='train')
            # test_ds = p3m.MattingDataset(split='test')
    else:
        print('Dataset not supported!')
        
    train_loader = DataLoader(train_ds, shuffle=True, num_workers=1, batch_size=batch_size)
    test_loader = DataLoader(test_ds, shuffle=True, num_workers=1, batch_size=batch_size)
    
    return train_loader, test_loader

def get_dataset_and_loader_with_split(dataset, batch_size, resolution, class_num=None, train_sample_num=None, test_sample_num=None, total_num=None, ratio=None, split='train'):
    loader = None
    if split == 'train':
        transform = get_training_transform(dataset, resolution)
    else:
        transform = get_test_transform(dataset, resolution)
    
    exist_datasets = ['cifar100', 'food101', 'imagenet', 'flowers102', 'cifar10', 'cub200', 'caltech256', 'dogs', 'sun397']
    assert dataset in exist_datasets

    if dataset == 'cifar10' and train_sample_num is None and total_num is not None:
        if split == 'train':
            ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        else:
            ds = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        if split == 'train':
            ds = Prototype(root=f'./data/{dataset}', transform=transform, split='train', class_num=class_num, sample_num=train_sample_num, total_num=total_num,  ratio=ratio)
        else:
            ds = Prototype(root=f'./data/{dataset}', transform=transform, split='test', class_num=class_num, sample_num=test_sample_num, total_num=total_num, ratio=ratio)
    
    loader = DataLoader(ds, shuffle=True, num_workers=4, batch_size=batch_size)
    return loader

def get_dataset_and_loader_with_flie(dataset, batch_size, resolution, file_path):
    # exist_datasets = ['cifar100', 'food101', 'imagenet', 'flowers102', 'cifar10', 'cub200', 'caltech256', 'dogs', 'sun397']
    # assert dataset in exist_datasets
    transform, target_transform = get_training_transform(dataset, resolution)
    ds = Prototype(root=f'./data/{dataset}', transform=transform, split='train', file_path=file_path)
    loader = DataLoader(ds, shuffle=True, num_workers=1, batch_size=batch_size)
    return loader

def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

def PCA_svd(X, k, center=True):
    n = X.size()[0]
    ones = torch.ones(n).view([n,1])
    h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
    H = torch.eye(n) - h
    H = H.cuda()
    X_center =  torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center)
    components  = v[:k].t()
    #explained_variance = torch.mul(s[:k], s[:k])/(n-1)
    return components


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
    

def general_class_file(class_file_path, save_path, class_num, sample_num):
    class_names = os.listdir(class_file_path)
    classes = []
    for class_name in class_names:
        if math.floor(len(os.listdir(os.path.join(class_file_path, class_name))) * 0.8) >= sample_num:
            classes.append(class_name)
        if len(classes) == class_num:
            break
    if len(classes) < class_num:
        print('Error: Not Enough images')
        exit(1)
    f = open(save_path, 'w')
    for n in classes:
        f.write(n + '\n')
    f.close()
    print('Success ' + save_path)

def general_class_file_temp():
    class_index = [2, 21, 38, 54, 55, 57, 67, 79, 83, 89, 95, 110, 115, 117, 119, 128, 131, 137, 139, 154, 161, 162, 165, 174, 179, 182, 203, 210, 243, 246, 269, 274, 277, 292, 303, 308, 309, 317, 324, 332]
    class_file_path = '/nfs4-p1/wjx/transferbility/experiment/data/DomainNet/clipart'
    save_path = '/nfs4-p1/wjx/transferbility/experiment/data/DomainNet/traintest/real_40_class'
    class_names = os.listdir(class_file_path)
    select_class_names = []
    for i in class_index:
        select_class_names.append(class_names[i])
    f = open(save_path, 'w')
    for n in select_class_names:
        f.write(n + '\n')
    f.close()
    print('Success ' + save_path)

def compute_pearson(Mat1, Mat2):
    return np.corrcoef(Mat1, Mat2)


def get_acc_from_log(log_path, save_path='/nfs4/wjx/transferbility/experiment/log/temp.log'):
    with open(log_path, 'r') as f:
        lines = f.readlines()
        acc = []
        for line in lines:
            if 'Accuracy:' in line:
                acc.append(float(line[-7:-1].strip()))
        
    f_w = open(save_path, 'a')
    f_w.write(log_path + '\n')
    f_w.write(str(acc))
    f_w.write('\n')
    f_w.close()

def norm(x):
    x = 200 * math.atan(x)
    x /= math.pi
    if x <= 0:
        x += 100
    return x

def train(epoch, model, train_loader, optimizer, loss_function, writer, device, warmup_scheduler=None):

    model.train()

    tqdm_train_loader = tqdm(train_loader, file=sys.stdout, position=0)
    acc = 0.0
    sample_num = 0
    for batch_index, (images, labels) in enumerate(tqdm_train_loader):
        if warmup_scheduler and epoch <= 1:
            warmup_scheduler.step()
        images = Variable(images)
        labels = Variable(labels)

        labels = labels.to(device)
        images = images.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        if type(outputs) == torchvision.models.inception.InceptionOutputs:
            outputs = outputs.logits
        elif type(outputs) == collections.OrderedDict:
            outputs = outputs['out']
        
        loss = loss_function(outputs, labels.long())
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(train_loader) + batch_index + 1
        sample_num += images.shape[0]

        acc += torch.sum((torch.argmax(outputs.data,1)) == labels.data)
        # last_layer = list(model.children())[-1]
        # for name, para in last_layer.named_parameters():
        #     if 'weight' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
        #     if 'bias' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        tqdm_train_loader.set_description('Training Epoch: {epoch}, Loss: {:0.4f}, LR: {:0.6f}, ACC: {:0.4f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            acc / sample_num,
            epoch=epoch,
            trained_samples=batch_index * 128 + len(images),
            total_samples=len(train_loader.dataset),
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

def train_seg(epoch, model, train_loader, optimizer, loss_function, writer,  num_classes, device):
    model.train()

    tqdm_train_loader = tqdm(train_loader, file=sys.stdout, position=0)
    acc = 0.0
    cm = ConfusionMatrix(num_classes=num_classes).to(device)
    for batch_index, (images, labels) in enumerate(tqdm_train_loader):
        images = Variable(images)
        labels = Variable(labels)

        labels = labels.to(device)
        images = images.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        if type(outputs) == torchvision.models.inception.InceptionOutputs:
            outputs = outputs.logits
        elif type(outputs) == collections.OrderedDict:
            outputs = outputs['out']
        
        loss = loss_function(outputs, labels.long())
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(train_loader) + batch_index + 1

        acc += torch.sum((torch.argmax(outputs.data,1)) == labels.data) / (480 * 320)
        cm.update(outputs=outputs, labels=labels)
        # last_layer = list(model.children())[-1]
        # for name, para in last_layer.named_parameters():
        #     if 'weight' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
        #     if 'bias' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        tqdm_train_loader.set_description('Training Epoch: {epoch}, Loss: {:0.4f}, LR: {:0.6f}, ACC: {:0.4f}, IoU: {:0.10f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            acc / len(train_loader.dataset),
            m_iou(cm.compute()),
            epoch=epoch,
            trained_samples=batch_index * 128 + len(images),
            total_samples=len(train_loader.dataset),
        ))
    
    
def train_mat(epoch, model, train_loader, optimizer, writer=None, device='cpu'):
    tqdm_train_loader = tqdm(train_loader, file=sys.stdout, position=0, desc='Fitting Decoder')
    blurer = GaussianBlurLayer(1, 3).to(device)
    semantic_scale=10.0
    detail_scale=10.0
    matte_scale=1.0
    
    # MAD
    mad_sum = 0.0
    
    # num_samples
    num_samples = 0
    
    for batch_index, (images, mattes, trimaps) in enumerate(tqdm_train_loader):
        batch_size = images.shape[0]
    
        images = Variable(images)
        trimaps = Variable(trimaps)
        mattes = Variable(mattes)

        trimaps = trimaps.to(device)
        images = images.to(device)
        mattes = mattes.to(device)

        # forward the model
        optimizer.zero_grad()
        pred_semantic, pred_detail, pred_matte = model(images, False)

        # calculate the boundary mask from the trimap
        boundaries = (trimaps == 0.0) + (trimaps == 1.0)

        # calculate the semantic loss
        gt_semantic = F.interpolate(mattes, scale_factor=1/16, mode='bilinear')
        gt_semantic = blurer(gt_semantic)
        semantic_loss = torch.mean(F.mse_loss(pred_semantic, gt_semantic))
        semantic_loss = semantic_scale * semantic_loss

        # calculate the detail loss
        pred_boundary_detail = torch.where(boundaries, trimaps, pred_detail)
        gt_detail = torch.where(boundaries, trimaps, mattes)
        detail_loss = torch.mean(F.l1_loss(pred_boundary_detail, gt_detail))
        detail_loss = detail_scale * detail_loss

        # calculate the matte loss
        pred_boundary_matte = torch.where(boundaries, trimaps, pred_matte)
        matte_l1_loss = F.l1_loss(pred_matte, mattes) + 4.0 * F.l1_loss(pred_boundary_matte, mattes)
        matte_compositional_loss = F.l1_loss(images * pred_matte, images * mattes) \
            + 4.0 * F.l1_loss(images * pred_boundary_matte, images * mattes)
        matte_loss = torch.mean(matte_l1_loss + matte_compositional_loss)
        matte_loss = matte_scale * matte_loss

        # calculate the final loss, backward the loss, and update the model 
        loss = semantic_loss + detail_loss + matte_loss
        loss.backward()
        optimizer.step()
        
        mad = torch.sum(torch.abs(pred_matte - mattes))    
        mad_sum += mad.item()
        num_samples += batch_size

        tqdm_train_loader.set_description('Training Epoch: {epoch}, Se_Loss: {:0.4f}, De_Loss: {:0.4f}, Ma_Loss: {:0.4f},  LR: {:0.6f}, MAD: {:0.4f}'.format(
            semantic_loss.item(),
            detail_loss.item(),
            matte_loss.item(),
            optimizer.param_groups[0]['lr'],
            mad_sum / num_samples,
            epoch=epoch,
            # trained_samples=batch_index * 128 + len(images),
            # total_samples=len(train_loader.dataset)
        ))


def eval_training(epoch, model, test_loader, loss_function, writer, device):
    model.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0 # top1 error

    with torch.no_grad():
        for (images, labels) in test_loader:
            images = Variable(images)
            labels = Variable(labels)

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)
            test_loss += loss.item()
            
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()

    #add informations to tensorboard
    writer.add_scalar('Test/Average_loss', test_loss / len(test_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(test_loader.dataset), epoch)

    return correct.float() / len(test_loader.dataset), test_loss / len(test_loader.dataset)

def eval_training_seg(epoch, model, test_loader, loss_function, num_classes, writer, device):
    model.eval()
    # torch.save(model.state_dict(), '123.pth')
    # exit(0)
    test_loss = 0.0 # cost function error
    cm = ConfusionMatrix(num_classes=num_classes).to(device)
    with torch.no_grad():
        for (images, labels) in test_loader:
            images = Variable(images)
            labels = Variable(labels)

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            if type(outputs) == torchvision.models.inception.InceptionOutputs:
                outputs = outputs.logits
            elif type(outputs) == collections.OrderedDict:
                outputs = outputs['out']

            loss = loss_function(outputs, labels)
            test_loss += loss.item()
        
            cm.update(outputs, labels)
    
    miou = m_iou(cm.compute())
    #add informations to tensorboard
    writer.add_scalar('Test/Average_loss', test_loss / len(test_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', miou, epoch)

    return miou, test_loss / len(test_loader.dataset)

def eval_training_mat(epoch, model, test_loader, writer, device):
    model.eval()

    test_loss = 0.0 # cost function error
    error_sum = 0.0 # miou

    with torch.no_grad():
        for (images, mattes, trimaps) in test_loader:
            images = Variable(images)
            mattes = Variable(mattes)

            images = images.to(device)
            mattes = mattes.to(device)

            _, _, pred_mattes = model(images, False)

            assert (len(pred_mattes.shape) == len(mattes.shape))
            # MAD 平均绝对差值
            error_sad = torch.sum(torch.abs(pred_mattes - mattes))    

            error_sum += error_sad
    
    #add informations to tensorboard
    writer.add_scalar('Test/Average_loss', test_loss / len(test_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', error_sum / len(test_loader.dataset), epoch)

    return error_sum / len(test_loader.dataset), test_loss / len(test_loader.dataset)

class ConfusionMatrix(object):
    def __init__(self, num_classes):
        super(ConfusionMatrix, self).__init__()
        self.num_classes = num_classes
        self.cm = torch.zeros((self.num_classes, self.num_classes))

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def update(self, outputs, labels):
        outputs = torch.argmax(outputs, dim=1)  # [B, C, H, W] -> [B, H, W]
        outputs = outputs.flatten()  # [B, H, W] -> [B*H*W]
        labels = labels.flatten()  # [B, H, W] -> [B*H*W]

        mask = (labels >= 0) & (labels < self.num_classes)
        inds = self.num_classes * labels[mask] + outputs[mask]
        self.cm += torch.bincount(inds, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)

    def compute(self):
        return self.cm

    def to(self, device):
        self.cm = self.cm.to(device)
        return self
    
def iou(cm):
    # IoU = TP / (TP + FP + FN)
    intersection = torch.diag(cm)
    union = torch.sum(cm, dim=1) + torch.sum(cm, dim=0) - torch.diag(cm)
    ious = intersection / union
    return ious


def dice(cm):
    # IoU =  2 TP / (2 TP + FP + FN)
    intersection = 2 * torch.diag(cm)
    union = torch.sum(cm, dim=1) + torch.sum(cm, dim=0)
    ious = intersection / union
    return ious


def m_iou(cm):
    miou = iou(cm).mean()
    return miou

# 中间层特征提取
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
        self.features = []

        for name, module in self.submodule.named_modules():
            if name in self.extracted_layers:
                module.register_forward_hook(self.hook)
        
    
    def hook(self, module, fea_in, fea_out):
        self.features.append(fea_in[0].detach())

    def forward(self, x):
        self.submodule(x)
    
    def output_and_clear(self):
        features = self.features
        self.features = []
        return features
    

class GaussianBlurLayer(nn.Module):
    """ Add Gaussian Blur to a 4D tensors
    This layer takes a 4D tensor of {N, C, H, W} as input.
    The Gaussian blur will be performed in given channel number (C) splitly.
    """

    def __init__(self, channels, kernel_size):
        """ 
        Arguments:
            channels (int): Channel for input tensor
            kernel_size (int): Size of the kernel used in blurring
        """

        super(GaussianBlurLayer, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 != 0

        self.op = nn.Sequential(
            nn.ReflectionPad2d(math.floor(self.kernel_size / 2)), 
            nn.Conv2d(channels, channels, self.kernel_size, 
                      stride=1, padding=0, bias=None, groups=channels)
        )

        self._init_kernel()

    def forward(self, x):
        """
        Arguments:
            x (torch.Tensor): input 4D tensor
        Returns:
            torch.Tensor: Blurred version of the input 
        """

        if not len(list(x.shape)) == 4:
            print('\'GaussianBlurLayer\' requires a 4D tensor as input\n')
            exit()
        elif not x.shape[1] == self.channels:
            print('In \'GaussianBlurLayer\', the required channel ({0}) is'
                  'not the same as input ({1})\n'.format(self.channels, x.shape[1]))
            exit()
            
        return self.op(x)
    
    def _init_kernel(self):
        sigma = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8

        n = np.zeros((self.kernel_size, self.kernel_size))
        i = math.floor(self.kernel_size / 2)
        n[i, i] = 1
        kernel = scipy.ndimage.gaussian_filter(n, sigma)

        for name, param in self.named_parameters():
            param.data.copy_(torch.from_numpy(kernel))


def mapping_values(values):
    values = np.array(values)
    R = np.max(values) - np.min(values)
    # print(R)
    mean = np.mean(values)
    # print(mean)
    f = lambda x: 50*(2.0 * np.arctan((x-mean) / R ) / np.pi  + 1.0)
    # f = lambda x : 2.0 * np.arctan((x-mean) / R ) / np.pi + 1
    t_values = []
    for v in values:
        t_values.append(f(v))
    print(t_values)

if __name__ == '__main__':
    from scipy.stats import pearsonr, spearmanr
    # # m-o logme
    
    # GT_mo = np.array([0.49579, 0.75063, 0.4246 ,0.3711, 0.44777 ,0.50548, 0.46083])

    # GT_om = np.array([0.49579, 0.4275, 0.6425, 0.9415, 0.905, 0.385, 0.67])

    # GT_ar = np.array([0.62174, 0.74684, 0.75063, 0.73168, 0.84962, 0.79612])

    # LogME_mo = np.array([1.2324,1.25075,1.23249,1.23172,1.23254,1.23271,1.23211])

    # LogME_om = np.array([1.57609,1.24703,1.39124,0.09124,1.00329,0.91179,1.03166])

    # LogME_ar = np.array([1.18072,1.24652,1.25081,1.24,1.26705,1.18318])

    # LEEP_mo = np.array([-3.81371,-3.25947,-3.68275,-4.8068,-4.41297,-4.08197,-4.15324])

    # LEEP_om = np.array([-2.81162,-3.19765,-1.44273,-1.88426,-2.74814,-3.15081,-1.10219])

    # LEEP_ar = np.array([-3.2922,-3.13516,-3.26159,-4.11717,-3.64939,-3.44816])

    # IDS_mo = np.array([-27.64733,-28.53157,-28.13547,-26.79247,-26.67806,-26.65203,-26.70478])

    # IDS_om = np.array([-28.80036,-28.51083,-28.92809,-28.64385,-28.61754,-28.57663,-28.60045])

    # cli, info, painting, real, quickdraw
    # GT_mo = np.array([0.3890000283718109, 0.18800000846385956, 0.37700000405311584 ,0.39400002360343933, 0.3110000193119049])

    # # cli, info, painting, sketch, quickdraw
    # GT_om = np.array([0.5480000376701355, 0.18000000715255737, 0.3790000081062317, 0.39400002360343933, 0.5250000357627869])

    # # mobile, resnet, incep, alex, swin_s, vit_b
    # GT_ar = np.array([0.46300002932548523, 0.39400002360343933, 0.4960000216960907, 0.27800002694129944, 0.3070000112056732, 0.1900000125169754])


    # LogME_mo = np.array([0.5739926387444876, 0.5519436149326741, 0.5723470405099306, 0.5908459123030685, 0.5588105688993563])

    # LogME_om = np.array([0.6343661211819408, 0.559709910247752, 0.625288303899985, 0.5928354807450826, 0.5887597434546897])

    # LogME_ar = np.array([0.5825989774763034, 0.598927455924812, 0.6121091685030811, 0.5626883751752962, 0.5517454814104927, 0.5364340513810401])

    # LEEP_mo = np.array([-3.089790105819702, -3.636406898498535, -3.107834577560425, -3.13215970993042, -3.4699504375457764])

    # LEEP_om = np.array([-2.6616179943084717, -3.435424566268921, -2.7635786533355713, -3.358710527420044, -3.126843214035034])

    # LEEP_ar = np.array([-2.832770586013794, -3.3340561389923096, -2.8627212047576904, -3.1184189319610596, -3.3261067867279053, -3.469856023788452])

    # IDS_mo = np.array([-26.346697248458863, -26.432270627975463, -27.556925579071045, -27.41812026023865, -23.758436377525328])

    # IDS_om = np.array([-28.569600093841554, -28.584445363998412, -28.77046439552307, -28.352270320892334, -28.489394954681398])
    # print(float(pearsonr(GT_mo, IDS_mo)[0]))
    # print(float(pearsonr(GT_mo, LogME_mo)[0]))
    # print(float(pearsonr(GT_mo, LEEP_mo)[0]))

    # print(float(pearsonr(GT_om, IDS_om)[0]))
    # print(float(pearsonr(GT_om, LogME_om)[0]))
    # print(float(pearsonr(GT_om, LEEP_om)[0]))

    # print(float(pearsonr(GT_ar, LogME_ar)[0]))
    # print(float(pearsonr(GT_ar, LEEP_ar)[0]))
    # mapping_values(IDS_mo)
    # mapping_values(LogME_mo)
    # mapping_values(LEEP_mo)


    from log.final.estimated_measures.classe import class_results
    from log.final.estimated_measures.domain import domain_results
    from log.final.estimated_measures.task import task_results
    from scipy.stats import pearsonr, spearmanr
    from scipy.stats.stats import kendalltau


    domain_multi_to_one = domain_results.multi_to_one
    domain_one_to_multi = domain_results.one_to_multi
    domain_diff_arch = domain_results.diff_arch

    class_multi_to_one = class_results.multi_to_one
    class_one_to_multi = class_results.one_to_multi
    class_diff_arch = class_results.diff_arch

    task_multi_to_one = task_results.multi_to_one

    dt = task_multi_to_one
    gt = np.array(dt['gt'])
    dt.pop('gt')
    for name, scores in dt.items():
        defen = {}
        scores = np.array(scores)
        defen['pearson'] = pearsonr(gt, scores)[0]
        defen['spearman'] = spearmanr(gt, scores)[0]
        defen['kendall'] = kendalltau(gt, scores)[0]
        dt[name] = defen
    
    print(dt)



    
