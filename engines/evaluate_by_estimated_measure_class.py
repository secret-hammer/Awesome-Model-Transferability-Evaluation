import argparse
import itertools
import os
import time
import sys 
import logging
import pathlib
from pathlib import Path
import gc
import math

import numpy as np
import torch
from torch import nn
import torch.optim as optim

import torchvision
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models

from torch.autograd import Variable

from tensorboardX import SummaryWriter
from tqdm.auto import tqdm

from torchsummary import summary
from sklearn.decomposition import PCA

from pytorch_pretrained_vit import ViT

import global_settings as settings
from global_settings import WEIGHT_PATH
from utils import get_dataset_and_loader, get_dataset_and_loader_with_flie, get_torchvision_network, get_network, load_weight, get_dataset_and_loader_with_flie, get_dataset_and_loader, FeatureExtractor
from metrics.metric_engine import evaluate_predicted_transferability

# 中间层特征提取
class FeatureExtractor_VIT(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor_VIT, self).__init__()
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
    
def prepare_kwargs_poss(
    feature_extractor,
    target_train_loader,
    device='cuda:0',
    target_data_ratio=1.0
):
    target_data_batches = None
    if target_data_ratio < 1.0:
        target_data_batches = math.ceil(target_data_ratio * len(target_train_loader.dataset) / target_train_loader.batch_size) - 1
        target_train_loader = itertools.islice(target_train_loader, 0, target_data_batches)
    
    features_tar_list = []
    labels_tar_list = []
    for images, labels in tqdm(target_train_loader):
        labels = labels.to(device)
        images = images.to(device)
        features = feature_extractor(images)
        if type(features) == torchvision.models.inception.InceptionOutputs:
            features = features.logits
        features_tar_list.append(features.detach())
        labels_tar_list.append(labels.detach())
        
    features_tar =features_tar_list[0]
    labels_tar = labels_tar_list[0]
    for feature in features_tar_list[1:]:
        features_tar = torch.cat((features_tar, feature), 0)
    for label in labels_tar_list[1:]:
        labels_tar = torch.cat((labels_tar, label), 0)
    
    features_tar = features_tar.view(features_tar.shape[0], -1)

    print(features_tar.shape)
    print(labels_tar.shape)
    kwargs = {
        'features_tar' : features_tar,
        'labels_tar' : labels_tar,
        'device' : device
    }
    return kwargs


# target_data_ratio source_data_ratio用于控制取用百分之多少的数据进行提取
def prepare_kwargs(
    feature_extractor,
    target_train_loader,
    source_train_loader=None,
    device='cuda:0',
    target_data_ratio=1.0,
    source_data_ratio=1.0
):
    target_data_batches = source_data_batches = None
    if target_data_ratio < 1.0:
        target_data_batches = math.ceil(target_data_ratio * len(target_train_loader.dataset) / target_train_loader.batch_size) - 1
        target_train_loader = itertools.islice(target_train_loader, 0, target_data_batches)
        print(target_data_batches)
    if source_data_ratio < 1.0 and source_train_loader is not None:
        source_data_batches = math.ceil(source_data_ratio * len(source_train_loader.dataset) / source_train_loader.batch_size) - 1
        source_train_loader = itertools.islice(source_train_loader, 0, source_data_batches)
        print(source_data_batches)
    
    features_src_list = []
    labels_src_list = []
    features_tar_list = []
    labels_tar_list = []
    for images, labels in tqdm(target_train_loader):
        labels = labels.to(device)
        images = images.to(device)
        feature_extractor(images)
        labels_tar_list.append(labels)

    features_tar_list = feature_extractor.output_and_clear()
    features_tar = features_tar_list[0]
    labels_tar = labels_tar_list[0]
    for feature in features_tar_list[1:]:
        features_tar = torch.cat((features_tar, feature), 0)
    for label in labels_tar_list[1:]:
        labels_tar = torch.cat((labels_tar, label), 0)
    
    features_tar = features_tar.view(features_tar.shape[0], -1)

    if source_train_loader is not None:
        for images, labels in tqdm(source_train_loader):
            labels = labels.to(device)
            images = images.to(device)
            feature_extractor(images)
            labels_src_list.append(labels)

        features_src_list = feature_extractor.output_and_clear()
        features_src =features_src_list[0]
        labels_src = labels_src_list[0]
        for feature in features_src_list[1:]:
            features_src = torch.cat((features_src, feature), 0)
        for label in labels_src_list[1:]:
            labels_src = torch.cat((labels_src, label), 0)
        
        features_src = features_src.view(features_src.shape[0], -1)
        print(features_src.shape)

    print(features_tar.shape)
    if source_train_loader is not None:
        kwargs = {
            'features_src' : features_src,
            'features_tar' : features_tar,
            'labels_src' : labels_src,
            'labels_tar' : labels_tar,
            'device' : device
        }
    else:
        kwargs = {
            'features_tar' : features_tar,
            'labels_tar' : labels_tar,
            'device' : device
        }
    return kwargs


weight_paths = {
    'sun397' : '/nfs4/wjx/transferbility/experiment/checkpoints/final/pre-training/resnet18/sun397/2023-07-24T12:33:10.894252/resnet18-60-best.pth',
    'imagenet' : None,
    'caltech256' : '/nfs4/wjx/transferbility/experiment/checkpoints/final/pre-training/resnet18/caltech256/2023-07-24T07:27:49.711921/resnet18-73-best.pth',
    'cifar10' : '/nfs4/wjx/transferbility/experiment/checkpoints/final/pre-training/resnet18/cifar10/2023-07-24T08:32:09.510653/resnet18-97-best.pth',
    'flowers102' : '/nfs4/wjx/transferbility/experiment/checkpoints/final/pre-training/resnet18/flowers102/2023-07-25T10:45:12.664932/resnet18-134-best.pth',
    'food101' : '/nfs4/wjx/transferbility/experiment/checkpoints/final/pre-training/resnet18/food101/2023-07-24T12:34:14.486441/resnet18-62-best.pth',
    'dogs' : '/nfs4/wjx/transferbility/experiment/checkpoints/final/pre-training/resnet18/dogs/2023-07-24T07:35:46.362153/resnet18-29-best.pth' 
}

# one-to-multi
class_nums = {
    'sun397' : 397,
    'imagenet' : 1000,
    'caltech256' : 257,
    'cifar10' : 10,
    'flowers102' : 102,
    'food101' : 101,
    'dogs' : 120,
    'cub200' : 200
}

def multi_source_to_target(args):
    scores = dict.fromkeys(weight_paths.keys(), None)
    logging.info('Multi to one')
    metric = args.metric
    print(metric)

    # 需要公开模型的在这里标注
    if metric == 'ids' or metric == 'emd':
        model = ViT('B_16', pretrained=True).to(args.device).eval()
        feature_extractor = FeatureExtractor_VIT(model, extracted_layers=['fc'])
    
    if metric == 'ids':
        target_train_loader = get_dataset_and_loader_with_flie(
            dataset = 'cub200',
            batch_size=32,
            resolution=224,
            file_path='/nfs4/wjx/transferbility/experiment/data/cub200/meta/multi_one_ids_1000'
        )
    else:
        target_train_loader = get_dataset_and_loader_with_flie(
            dataset = 'cub200',
            batch_size=32,
            resolution=224,
            file_path='/nfs4/wjx/transferbility/experiment/data/cub200/meta/train_200_0.8'
        )

    for source_dataset, weight_path in weight_paths.items():
        print(source_dataset)
        source_train_loader = None
        if args.need_source:
            if metric == 'ids':
                file_path = f'/nfs4/wjx/transferbility/experiment/data/{source_dataset}/meta/multi_one_ids_1000'
                source_train_loader = get_dataset_and_loader_with_flie(
                    dataset=source_dataset,
                    resolution=224,
                    batch_size=32,
                    file_path = file_path
                )
            else:
                file_path = f'/nfs4/wjx/transferbility/experiment/data/{source_dataset}/meta/train_{class_nums[source_dataset]}_0.8' if source_dataset != 'imagenet' else f'/nfs4/wjx/transferbility/experiment/data/{source_dataset}/meta/train_1000_10000_0.8'
                source_train_loader = get_dataset_and_loader_with_flie(
                    dataset=source_dataset,
                    resolution=224,
                    batch_size=32,
                    file_path = file_path
                )
        class_num = class_nums[source_dataset]
        if metric != 'ids' and metric != 'emd':
            if weight_path is None:
                model = get_torchvision_network('resnet18', class_num=class_num, pretrain=True).to(args.device).eval()
            else:   
                model = get_torchvision_network('resnet18', class_num=class_num, pretrain=False).to(args.device).eval()
                load_weight(model, weight_path=weight_path, device=args.device)
        
        kwargs = None
        if args.feature == 'rep':
            if metric != 'ids' and metric != 'emd':
                feature_extractor = FeatureExtractor(model, extracted_layers=['fc'])
            kwargs = prepare_kwargs(
                feature_extractor=feature_extractor,
                target_train_loader=target_train_loader,
                source_train_loader=source_train_loader,
                target_data_ratio=args.target_data_ratio,
                source_data_ratio=args.source_data_ratio if source_dataset != 'imagenet' else 1.0,
                device=args.device
            )
        elif args.feature == 'pos':
            kwargs = prepare_kwargs_poss(
                feature_extractor=model, 
                target_train_loader=target_train_loader,
                target_data_ratio=args.target_data_ratio,
                device=args.device
            )

        score = evaluate_predicted_transferability(args.metric, 512,  **kwargs)
        scores[source_dataset] = score
        del source_train_loader
        del kwargs
        torch.cuda.empty_cache()
        gc.collect()

    
    return scores
    
    
def one_to_multi_target(args):
    logging.info('One to multi')
    target_datasets = ['sun397', 'caltech256', 'cifar10', 'cub200', 'food101', 'dogs', 'flowers102']
    scores = dict.fromkeys(target_datasets, None)
    metric = args.metric
    print(metric)
    
    # 需要公开模型的在这里标注
    if metric == 'ids':
        model = ViT('B_16', pretrained=True).to(args.device).eval()
        feature_extractor = FeatureExtractor_VIT(model, extracted_layers=['fc'])
    else:
        model = get_torchvision_network('resnet18', 1000, pretrain=True).to(args.device).eval()
        if args.feature == 'rep':
            feature_extractor = FeatureExtractor(model, extracted_layers=['fc'])
        elif args.feature == 'pos':
            feature_extractor = model
        
    source_train_loader = None
    if args.need_source:
        if metric == 'ids':
            file_path = '/nfs4/wjx/transferbility/experiment/data/imagenet/meta/multi_one_ids_1000'
            source_train_loader = get_dataset_and_loader_with_flie(
                dataset = 'real',
                batch_size=32,
                resolution=224,
                file_path=file_path
            )
        else:
            file_path = '/nfs4/wjx/transferbility/experiment/data/imagenet/meta/train_1000_10000_0.8'
            source_train_loader = get_dataset_and_loader_with_flie(
                dataset = 'real',
                batch_size=32,
                resolution=224,
                file_path=file_path
            )
    
    source_feature = None
    source_label = None
    for target_dataset in target_datasets:
        print(target_dataset)
        if metric == 'ids':
            file_path = f'/nfs4/wjx/transferbility/experiment/data/{target_dataset}/meta/one_multi_ids_1000'
            target_train_loader = get_dataset_and_loader_with_flie(
                dataset=target_dataset,
                batch_size=32,
                resolution=224,
                file_path = file_path
            )
        else:
            target_train_loader = get_dataset_and_loader_with_flie(
                dataset=target_dataset,
                resolution=224,
                batch_size=32,
                file_path = f'/nfs4/wjx/transferbility/experiment/data/{target_dataset}/meta/train_{class_nums[target_dataset]}_2000_0.8'
            )
        
        kwargs = None
        if args.feature == 'rep':
            if args.need_source:
                if source_feature is None:
                    kwargs = prepare_kwargs(
                        feature_extractor=feature_extractor,
                        target_train_loader=target_train_loader,
                        source_train_loader=source_train_loader ,
                        target_data_ratio=args.target_data_ratio,
                        source_data_ratio=args.source_data_ratio,
                        device=args.device
                    )
                else:
                    kwargs = prepare_kwargs(
                        feature_extractor=feature_extractor,
                        target_train_loader=target_train_loader,
                        target_data_ratio=args.target_data_ratio,
                        device=args.device
                    )
                    kwargs['features_src'] = source_feature
                    kwargs['labels_src'] = source_label
            else:
                kwargs = prepare_kwargs(
                    feature_extractor=feature_extractor,
                    target_train_loader=target_train_loader,
                    target_data_ratio=args.target_data_ratio,
                    device=args.device
                )
        elif args.feature == 'pos':
            kwargs = prepare_kwargs_poss(
                feature_extractor=model, 
                target_train_loader=target_train_loader,
                target_data_ratio=args.target_data_ratio,
                device=args.device
            )
        
        if args.need_source and source_feature is None:
            source_feature = kwargs['features_src']
        if args.need_source and source_label is None:
            source_label = kwargs['labels_src']
        score = evaluate_predicted_transferability(metric, 512, **kwargs)
            
        scores[target_dataset] = score
        del target_train_loader
        del kwargs
        gc.collect()
        torch.cuda.empty_cache()
        
    return scores


def diff_arch(args):
    archs = ['mobilenet_v2', 'resnet18', 'inception_v3', 'alexnet', 'swin_s', 'vit_b_16']
    scores= dict.fromkeys(archs, None)
    arch_feature_layer = {
        'mobilenet_v2' : 'classifier.1', 
        'alexnet' : 'classifier.4', 
        'resnet18' : 'fc', 
        'inception_v3' : 'fc', 
        'swin_s' : 'head', 
        'vit_b_16' : 'heads.head'
    }
    target_train_loader = get_dataset_and_loader_with_flie(
        dataset = 'cub200',
        batch_size=32,
        resolution=224,
        file_path='/nfs4/wjx/transferbility/experiment/data/cub200/meta/train_200_0.8'
    )
    target_train_loader_inception = get_dataset_and_loader_with_flie(
        dataset = 'cub200',
        batch_size=32,
        resolution=299,
        file_path='/nfs4/wjx/transferbility/experiment/data/cub200/meta/train_200_0.8'
    )
    
    source_train_loader = None
    if args.need_source:
        if args.metric == 'ids':
            file_path = '/nfs4/wjx/transferbility/experiment/data/imagenet/meta/multi_one_ids_1000'
            source_train_loader = get_dataset_and_loader_with_flie(
                dataset = 'real',
                batch_size=32,
                resolution=224,
                file_path=file_path
            )
        else:
            file_path = '/nfs4/wjx/transferbility/experiment/data/imagenet/meta/train_1000_10000_0.8'
            source_train_loader = get_dataset_and_loader_with_flie(
                dataset = 'real',
                batch_size=32,
                resolution=224,
                file_path=file_path
            )

    metric = args.metric
    for arch in archs:
        print(arch)
        model = get_torchvision_network(arch, 1000, pretrain=True).to(args.device).eval()
        
        if args.feature == 'rep':
            feature_extractor = FeatureExtractor(model, extracted_layers=[arch_feature_layer[arch]])
            kwargs = prepare_kwargs(
                feature_extractor=feature_extractor,
                target_train_loader=target_train_loader if arch != 'inception_v3' else target_train_loader_inception,
                source_train_loader=source_train_loader,
                target_data_ratio=args.target_data_ratio,
                source_data_ratio=1.0,
                device=args.device
            )
        elif args.feature == 'pos':
            kwargs = prepare_kwargs_poss(
                feature_extractor=model, 
                target_train_loader=target_train_loader if arch != 'inception_v3' else target_train_loader_inception,
                target_data_ratio=args.target_data_ratio,
                device=args.device
            )
            
        score = evaluate_predicted_transferability(metric, 512, **kwargs)
        scores[arch] = score
        del kwargs
        gc.collect()
        torch.cuda.empty_cache()
    
    return scores
        
                

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str)
parser.add_argument('--metric', type=str)
# 在该实验中，need_source 固定源域 probe set size = 1000
# emd 指标由于其特殊性，我们单独进行测试
parser.add_argument('--need_source', action='store_true')
parser.add_argument('--source_data_ratio', type=float, default=1.0)
parser.add_argument('--target_data_ratio', type=float, default=1.0)
parser.add_argument('--feature', type=str)
parser.add_argument('--gpu', type=int, default=0)
if __name__ == '__main__':
    #设置随机种子
    seed=10
    torch.manual_seed(seed)
    args = parser.parse_args()
    args.log_path = f'/nfs4/wjx/transferbility/experiment/log/final/estimated_measures/class/{args.mode}/process.log'
    path_list, _ = os.path.split(args.log_path)
    if not os.path.exists(path_list):
        os.makedirs(path_list)
    # 配置日志格式
    logging.basicConfig(filename=args.log_path, level=logging.INFO)
    logging.info('')
    logging.info(args.__dict__)

    args.device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")
    if args.mode == 'multi_to_one':
        scores = multi_source_to_target(args)
    elif args.mode == 'one_to_multi':
        scores = one_to_multi_target(args)
    elif args.mode == 'diff_arch':
        scores = diff_arch(args)
    else:
        print('Wrong mode!')
    
    logging.info('%' + args.metric +'%')
    logging.info(scores)
    logging.info(scores.values())