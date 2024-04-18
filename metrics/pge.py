import argparse
import os
import sys 
import pathlib
from pathlib import Path
import logging

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
from utils import get_network, get_pretrain_network, get_training_transform, get_test_transform, get_torchvision_network
import global_settings as settings

# 在输入的模型本身叠加梯度（初始值被忽略）
# 返回值：model.state_dict()
def pge(model, model_name, data_loader, dataset_name, criterion, start_epoch=1, end_epoch=1, save_path=None):

    count = 0
    for epoch in range(start_epoch,end_epoch+1):
        #train
        tqdm_data_loader = tqdm(data_loader, file=sys.stdout, position=0)
        for i, (x, lables) in enumerate(tqdm_data_loader):
            new_model = ''
            if dataset_name in ['imagenet50', 'clatech256', 'food101']:
                new_model, _ = get_torchvision_network(model_name)
            else:
                new_model = get_network(model_name)

            num_classes = settings.NUM_CLASSES[dataset_name]
            if model_name == 'mobilenet_v3_small':
                if dataset_name in ['imagenet50', 'clatech256', 'food101']:
                    feature_dim = model.classifier[3].in_features
                    new_model.classifier[3] = nn.Linear(feature_dim, num_classes).cuda()
                else:
                    feature_dim = model.classifier[1].in_features
                    new_model.classifier[1] = nn.Linear(feature_dim, num_classes).cuda()
            elif model_name == 'inception_v3':
                Aux_feature_dim = model.AuxLogits.fc.in_features
                feature_dim = model.fc.in_features
                new_model.AuxLogits.fc = nn.Linear(Aux_feature_dim, num_classes).cuda()
                new_model.fc = nn.Linear(feature_dim, num_classes).cuda()
            elif model_name == 'vgg16':
                feature_dim = model.classifier[6].in_features
                new_model.classifier[6] = nn.Linear(feature_dim, num_classes).cuda()
            else:
                fc = getattr(new_model, settings.CLASSIFICATION_HEAD_NAME[model_name])
                feature_dim = fc.in_features
                setattr(new_model, settings.CLASSIFICATION_HEAD_NAME[model_name], nn.Linear(feature_dim, num_classes).cuda())

            x = Variable(x).cuda()
            lables = Variable(lables).cuda()

            new_model.zero_grad()
            preds = new_model(x)
            loss = criterion(preds, lables)
            loss.backward()

            # copy grads from new model to model
            for p,new_p in zip(model.parameters(),new_model.parameters()):
                p.data = (p.data*count + new_p.grad)/(count+1)
            count += 1

            tqdm_data_loader.set_description('Epoch {:d} | Batch {:d}/{:d} | Count {:d}'.format(epoch, i, len(data_loader), count))
        torch.cuda.empty_cache()
            
    torch.save({'epoch':epoch, 'state':model.state_dict()}, save_path)
    # return model.state_dict()


def get_dis(parameters1, parameters2):
    smi_count = 0
    smi_sum = 0
    for (k1,v1), (k2,v2) in zip(parameters1.items(),parameters2.items()):
        if 'fc' not in k1 and 'fc' not in k2 and 'classifier' not in k1 and 'classifier' not in k2 and 'running_mean'not in k1 and 'running_mean' not in k2 and 'running_var' not in k1 and 'running_var' not in k2 and 'num_batches_tracked' not in k1 and 'num_batches_tracked' not in k2 and 'mask' not in k1 and 'mask' not in k2 and 'mlp_head'not in k1 and 'mlp_head' not in k2 :
            assert k1 == k2
            w1 = v1.flatten()
            w2 = v2.flatten()
            if smi_count == 0:
                param1 = w1
                param2 = w2
            else:
                param1 = torch.cat([param1, w1], 0)
                param2 = torch.cat([param2, w2], 0)


            smi_count += 1

    a1 = torch.norm(param1)
    a2 = torch.norm(param2)
    similarity = torch.norm(param1-param2)/(a1*a2)
    
    return similarity.item()


