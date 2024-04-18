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
import torchvision.models as models

from tqdm.auto import tqdm


from models.encoder import get_resnet

from collections import OrderedDict
from pytorch_pretrained_vit import ViT

from utils import get_dataset_and_loader, get_dataset_and_loader_with_flie, get_torchvision_network, get_network, load_weight, get_dataset_and_loader_with_flie, get_dataset_and_loader, FeatureExtractor
from metrics.metric_engine import evaluate_predicted_transferability
# from multi_task.detection.coco_utils import get_coco
# import multi_task.detection.presets as presets

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

# target_data_ratio source_data_ratio用于控制取用百分之多少的数据进行提取
def prepare_kwargs_middle_level(
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
    
    down_sample = nn.AdaptiveAvgPool2d((1,1))
    features_src_list = []
    labels_src_list = []
    features_tar_list = []
    labels_tar_list = []
    for images, labels in tqdm(target_train_loader):
        labels = labels.to(device)
        images = images.to(device)
        features = feature_extractor(images)
        if type(features) == torchvision.models.inception.InceptionOutputs:
            features = features.logits
        
        features = torch.flatten(down_sample(features), 1)
        features_tar_list.append(features.detach())
        labels_tar_list.append(labels.detach())

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
            features = feature_extractor(images)
            if type(features) == torchvision.models.inception.InceptionOutputs:
                features = features.logits
            
            features = torch.flatten(down_sample(features), 1)
            features_src_list.append(features.detach())
            labels_src_list.append(labels.detach())

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

def multi_source_to_target(args):
    tasks = ['cls', 'seg', 'keypoint', 'mat', 'det']
    scores = dict.fromkeys(tasks, None)
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

    for task in tasks:
        print(task)
        source_train_loader = None
        if args.need_source:
            source_train_loader = get_source_loader(task)
        
        if metric != 'ids' and metric != 'emd':
            encoder = get_task_encoder(task)
            encoder.to(args.device).eval()
        
        if metric == 'ids' or metric == 'emd':
            kwargs = prepare_kwargs_middle_level(
                feature_extractor=feature_extractor,
                target_train_loader=target_train_loader,
                source_train_loader=source_train_loader,
                target_data_ratio=args.target_data_ratio,
                source_data_ratio=args.source_data_ratio,
                device=args.device
            )
        else:
            kwargs = prepare_kwargs(
                feature_extractor=encoder,
                target_train_loader=target_train_loader,
                source_train_loader=source_train_loader,
                target_data_ratio=args.target_data_ratio,
                source_data_ratio=args.source_data_ratio,
                device=args.device
            )   

        score = evaluate_predicted_transferability(args.metric, 512,  **kwargs)
        scores[task] = score
        del source_train_loader
        del kwargs
        torch.cuda.empty_cache()
        gc.collect()
    return scores

def get_dataset(is_train, args):
    image_set = "train" if is_train else "val"
    num_classes, mode = {"coco": (91, "instances"), "coco_kp": (2, "person_keypoints")}[args.dataset]
    with_masks = "mask" in args.model
    ds = get_coco(
        root=args.data_path,
        image_set=image_set,
        transforms=get_transform(is_train, args),
        mode=mode,
        use_v2=args.use_v2,
        with_masks=with_masks,
    )
    return ds, num_classes


def get_transform(is_train, args):
    if is_train:
        return presets.DetectionPresetTrain(
            data_augmentation=args.data_augmentation, backend=args.backend, use_v2=args.use_v2
        )
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()
        return lambda img, target: (trans(img), target)
    else:
        return presets.DetectionPresetEval(backend=args.backend, use_v2=args.use_v2)


def get_source_loader(task):
    if task == 'cls':
        file_path = f'/nfs4/wjx/transferbility/experiment/data/imagenet/meta/train_1000_10000_0.8'
        source_train_loader = get_dataset_and_loader_with_flie(
            dataset='imagenet',
            resolution=224,
            batch_size=32,
            file_path = file_path
        )
    elif task == 'seg':
        source_train_loader, _ = get_dataset_and_loader(
                                    dataset='voc', 
                                    resolution=224,
                                    batch_size=32
                                )
    # 从 https://github.com/pytorch/vision/blob/main/references/detection/ 获得的
    elif task == 'keypoint' or task == 'det':
        argss = {}
        argss['dataset'] = 'coco_kp' if task == 'keypoint' else 'coco'
        argss['data_path'] = "/datasets/COCO2017/"
        argss['use_v2'] = False
        argss['data_augmentation'] = 'hflip'
        argss['backend'] = 'tensor'
        ds = get_dataset(is_train=True, args=argss)
        source_train_loader = torchvision.utils.data.DataLoader(ds, shuffle=True, num_workers=1, batch_size=32)
    elif task == 'mat':
        source_train_loader, _ = get_dataset_and_loader(
                                    dataset='p3m', 
                                    resolution=224,
                                    batch_size=32
                                )
    return source_train_loader
        
def get_task_encoder(task):
    encoder = None
    if task == 'cls':
        encoder = get_resnet(num_layers=50)
        weight = torch.load('/home/wjx/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth')
        encoder.load_state_dict(weight, strict=False)
        # from torchsummary import summary
        # summary(encoder, input_size=[(3, 224 ,224)], batch_size=128, device="cpu")
        # exit(0)
    elif task == 'seg':
        encoder = get_resnet(num_layers=50)
        model = torchvision.models.segmentation.fcn_resnet50(progress=True, num_classes=21)
        weight = torch.load('/nfs4/wjx/transferbility/experiment/checkpoints/final/pre-training/resnet50_fcn/voc/2023-09-26T05:05:41.968410/fcn_resnet50-146-best.pth')
        model.load_state_dict(weight, strict=True)
        module_dict = model.backbone
        # 重命名
        weights = OrderedDict()
        for k, v in module_dict.items():
            root_layer_name = k + '.'
            for layer_name, t in v.state_dict().items():
                weights[root_layer_name+layer_name] = t
        encoder.load_state_dict(weights, strict=True)
        
    elif task == 'keypoint':
        encoder = get_resnet(num_layers=50)
        from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
        model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
        module_dict = model.backbone.body
        weights = OrderedDict()
        for k, v in module_dict.items():
            root_layer_name = k + '.'
            for layer_name, t in v.state_dict().items():
                weights[root_layer_name+layer_name] = t
        encoder.load_state_dict(weights, strict=True)
    
    elif task == 'mat':
        encoder = get_resnet(num_layers=50)
        from models.modnet import MODNet
        model = MODNet(hr_channels=32, backbone_arch='resnet50', backbone_pretrained=True)
        weight = torch.load('/nfs4/wjx/transferbility/experiment/checkpoints/final/pre-training/modnet_resnet50/p3m/2023-09-26T07:35:05.730578/modnet_resnet50-168-best.pth')
        model.load_state_dict(weight, strict=True)
        modnet_encoder = model.backbone
        
        modnet_weight = modnet_encoder.state_dict()
        weights = OrderedDict()

        for k, v in modnet_weight.items():
            if 'ch_down' in k:
                continue
            nk = k.replace('model.', '')
            weights[nk] = v

        encoder.load_state_dict(weights, strict=True)
        
    elif task == 'det':
        encoder = get_resnet(num_layers=50)
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
        module_dict = model.backbone.body
        weights = OrderedDict()
        for k, v in module_dict.items():
            root_layer_name = k + '.'
            for layer_name, t in v.state_dict().items():
                weights[root_layer_name+layer_name] = t
        encoder.load_state_dict(weights, strict=True)
    
    return encoder

parser = argparse.ArgumentParser()
parser.add_argument('--metric', type=str)
# 在该实验中，need_source 固定源域 probe set size = 1000
# emd 指标由于其特殊性，我们单独进行测试
parser.add_argument('--need_source', action='store_true')
parser.add_argument('--source_data_ratio', type=float, default=1.0)
parser.add_argument('--target_data_ratio', type=float, default=1.0)
parser.add_argument('--gpu', type=int, default=0)
if __name__ == '__main__':
    #设置随机种子
    seed=10
    torch.manual_seed(seed)
    args = parser.parse_args()
    args.log_path = f'/nfs4/wjx/transferbility/experiment/log/final/estimated_measures/task/multi_to_one/process.log'
    path_list, _ = os.path.split(args.log_path)
    if not os.path.exists(path_list):
        os.makedirs(path_list)
    # 配置日志格式
    logging.basicConfig(filename=args.log_path, level=logging.INFO)
    logging.info('')
    logging.info(args.__dict__)

    args.device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")
    scores = multi_source_to_target(args)
    
    logging.info('%' + args.metric +'%')
    logging.info(scores)
    logging.info(scores.values())
    

