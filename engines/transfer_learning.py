import argparse
import os
import sys 
import logging
import pathlib
from pathlib import Path

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

import global_settings as settings
from utils import get_torchvision_network, get_network, load_weight, load_weight_till_target_layer, get_dataset_and_loader, train, eval_training, replace_new_classifier

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--pre_class_num', type=int)
parser.add_argument('--class_num', type=int)
parser.add_argument('--sample_num', type=int, default=None)
parser.add_argument('--total_num', type=int, default=None)
parser.add_argument('--ratio', type=float, default=None)
parser.add_argument('--arch', type=str)
parser.add_argument('--resolution', type=int)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--imagenet_pretrain', action='store_true', default=False)
parser.add_argument('--weight_path', type=str, default=None)
# parser.add_argument('--resume',action='store_true', default=False) # resume表示预训练没有完成的模型进一步训练
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--choose_layer', type=int, default=None)
parser.add_argument('--tensorboard_path', type=str)
parser.add_argument('--checkpoint_path', type=str)
parser.add_argument('--log_path', type=str)
parser.add_argument('--checkpoint_save_step', type=int, default=5)
parser.add_argument('--end_epoch', type=str, default=200)
parser.add_argument('--gpu', type=int, default=0)
# parser.add_argument('--gpu_id', type=str, default=0)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
#预训练
def transfer_learning(args):
    #获取数据集和dataloader
    if args.sample_num is not None:
        train_loader, test_loader = get_dataset_and_loader(
            dataset=args.dataset, 
            resolution=args.resolution,
            batch_size=args.bs, 
            class_num=args.class_num, 
            train_sample_num=args.sample_num,
            test_sample_num=20
        )
    elif args.ratio is not None:
        if args.total_num is None:
            train_loader, test_loader = get_dataset_and_loader(
                dataset=args.dataset, 
                resolution=args.resolution,
                batch_size=args.bs, 
                class_num=args.class_num, 
                ratio=args.ratio
            )
        else:
            train_loader, test_loader = get_dataset_and_loader(
                dataset=args.dataset, 
                resolution=args.resolution,
                batch_size=args.bs, 
                class_num=args.class_num,
                total_num=args.total_num, 
                ratio=args.ratio
            )
    else:
        print('Error: No sample num nor ratio was given!')


    logging.info(f'Train sample num: {len(train_loader.dataset)}' )
    logging.info(f'Test sample num: {len(test_loader.dataset)}' )
    start_epoch = 1
    #获取模型
    if args.arch == 'vgg11' or args.arch == 'vgg13' or args.arch == 'vgg16' or args.arch == 'vgg11_small' or args.arch == 'vgg11_tiny':
        model = get_network(args.arch, args.pre_class_num).to(args.device)
        # model = get_torchvision_network(args.arch, class_num=args.class_num, pretrain=False).to(args.device)
        # summary(model, input_size=[(3, 32 ,32)], batch_size=128, device="cuda")
    else:
        print(args.imagenet_pretrain)
        model = get_torchvision_network(args.arch, class_num=args.pre_class_num, pretrain=args.imagenet_pretrain).to(args.device)

    
    if not args.imagenet_pretrain:
        if not args.choose_layer:
            load_weight(model, args.weight_path, args.device)
            torch.cuda.empty_cache()
        else:
            load_weight_till_target_layer(model, args.weight_path, args.choose_layer)

    
    replace_new_classifier(
        model=model, 
        arch=args.arch, 
        class_num=args.class_num,
        device=args.device
    )
    # print(model)
    
    #预训练参数
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # train_scheduler = optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1)
    # train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.end_epoch - start_epoch + 1)
    train_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer, 
        milestones=[50, 100], 
        gamma=0.1,
        last_epoch=-1
    )
    #tensorboard设置
    writer = SummaryWriter(log_dir=os.path.join(args.tensorboard_path, settings.TIME_NOW))

    #预训练checkpoint设置
    checkpoint_path = os.path.join(args.checkpoint_path, settings.TIME_NOW)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{arch}-{epoch}-{type}.pth')
    #训练
    best_acc = 0.0
    best_epoch = 0
    for epoch in range(start_epoch, args.end_epoch + 1):
        train(epoch, model, train_loader, optimizer, loss_function, writer, args.device)
        acc, test_loss = eval_training(epoch, model, test_loader, loss_function, writer, args.device)

        logging.info('Epoch {:d}, Test set: LR: {:.8f},  Average loss: {:.4f}, Accuracy: {:.4f}'.format(
            epoch, optimizer.param_groups[0]['lr'], test_loss, acc 
        ))
        
        #start to save best performance model after learning rate decay to 0.01 
        if best_acc < acc:
            torch.save(model.state_dict(), checkpoint_path.format(arch=args.arch, epoch=epoch, type='best'))
            best_acc = acc
            best_epoch = epoch
            continue

        if not epoch % args.checkpoint_save_step:
            torch.save(model.state_dict(), checkpoint_path.format(arch=args.arch, epoch=epoch, type='regular'))

        train_scheduler.step()
    
    writer.close()
    logging.info(f'Best Epoch: {best_epoch}')
    logging.info(f'Best Acc: {best_acc}')


if __name__ == '__main__':

    #设置随机种子
    seed=10
    torch.manual_seed(seed)
    args = parser.parse_args()
    
    path_list, _ = os.path.split(args.log_path)
    
    if not os.path.exists(path_list):
        os.makedirs(path_list)
    
    
    print(torch.cuda.device_count())
    args.device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")
    print(args.device)
    # 配置日志格式
    logging.basicConfig(filename=args.log_path, level=logging.INFO)
    logging.info('Description: 迁移学习日志，参数如下')
    logging.info(args.__dict__)
    
    transfer_learning(args)
        


    