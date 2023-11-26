# -*- coding: utf-8 -*-
import math
import torch
from functools import reduce
from operator import mul
import random
import json
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import os
import sys
import time
import logging
import numpy as np
import torch.nn.functional as F
from pytorch_metric_learning import distances
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from prefetch_generator import BackgroundGenerator, background
from torch.autograd import Variable
term_width = 100
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def set_parameters(mod, grad_list):
    offset = 0
    for name, param in mod.named_parameters():
        weight_flat_size = reduce(mul, param.size(), 1)
        param.data = grad_list[offset:offset + weight_flat_size].data.view(*param.size())
        offset += weight_flat_size

def get_flat_parameters(m):
    lst_param = []

    for name, param in m.named_parameters():
        lst_param.append(param.view(1, -1))

    param = torch.cat(lst_param, 1).view(-1)
    return param

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

class MyDataSet(Dataset):
    """自定义数据集"""
    def __init__(self,
                 root_dir: str,
                 csv_name: str,
                 json_path: str,
                 transform=None):
        images_dir = os.path.join(root_dir, "images")
        assert os.path.exists(images_dir), "dir:'{}' not found.".format(images_dir)

        assert os.path.exists(json_path), "file:'{}' not found.".format(json_path)
        self.label_dict = json.load(open(json_path, "r"))

        csv_path = os.path.join(root_dir, csv_name)
        assert os.path.exists(csv_path), "file:'{}' not found.".format(csv_path)
        csv_data = pd.read_csv(csv_path)
        self.total_num = csv_data.shape[0]
        self.img_paths = [os.path.join(images_dir, i)for i in csv_data["filename"].values]
        self.img_label = [self.label_dict[i][0] for i in csv_data["label"].values]
        self.labels = set(csv_data["label"].values)

        self.transform = transform

    def __len__(self):
        return self.total_num

    def __getitem__(self, item):
        img = Image.open(self.img_paths[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.img_paths[item]))
        label = self.img_label[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

def getPrototype(data, label, num_category):
    '''
    :param data: 验证数据，size：[batch, datashape]; label: 数据标签; num_category: 类别数量
    :return: 每个类别的原型
    '''

    dict_category = {}
    for category in range(num_category):
        dict_category[category] = []

    for idx, y in enumerate(label):
        y = y.item()
        dict_category[y].append(data[idx].view(1, -1))

    dict_prototype = {}
    for category in range(num_category):
        if len(dict_category[category]) != 0:
            dict_prototype[category] = torch.cat(dict_category[category], 0)
        else:
            dict_prototype[category] = None

    return dict_prototype

def reweight(traindata, train_label, val_prototype):
    bs = traindata.shape[0]
    lst_prototype = []
    for y in train_label:
        y = y.item()
        lst_prototype.append(val_prototype[y].view(1, -1))

    prototype = torch.cat(lst_prototype, 0)
    snr_dis = distances.SNRDistance()
    cos = snr_dis(traindata.view(bs, -1), prototype.view(bs, -1))
    cos = (torch.sum(cos, -1) / torch.sum(cos)).view(-1, 1)
    # cos = F.cosine_similarity(traindata.view(bs, -1), prototype.view(bs, -1), dim=1)
    return cos.data

def imshow(img):
    print(img.shape)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def detach_diff_opt(state):
    for key in state.keys():
        state[key]['momentum_buffer'] = Variable(state[key]['momentum_buffer'])

    return state


if __name__ == '__main__':
    torchvision.datasets.ImageNet(root='/home/hzy/space/', split='train', download=True)
    torchvision.datasets.ImageNet(root='/home/hzy/space/', split='val', download=True)

    # normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    #                                  std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize
    # ])
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     normalize
    # ])
    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    #
    # train_data, val_data = torch.utils.data.random_split(trainset, [25000, 25000])
    # t_loader = torch.utils.data.DataLoader(train_data, batch_size=25, shuffle=True)
    # v_loader = torch.utils.data.DataLoader(val_data, batch_size=100, shuffle=True)
    # v_total  = torch.utils.data.DataLoader(val_data, batch_size=25000, shuffle=True)
    # num_category = 10
    #
    # dict_prototype = {}
    # for category in range(num_category):
    #     dict_prototype[category] = []
    #
    # for idx, (x, y) in enumerate(v_loader):
    #     bs, c, w, h = x.shape
    #     dict_prototype_bs = getPrototype(x, y, num_category)
    #     for cls in range(num_category):
    #         if dict_prototype_bs[cls] != None:
    #             dict_prototype[cls].append(dict_prototype_bs[cls])
    # for category in range(num_category):
    #     dict_prototype[category] = torch.mean(torch.cat(dict_prototype[category], 0), 0).view(1, -1)
    #
    # for category in range(num_category):
    #     print(category, dict_prototype[category].shape)
    #
    # for idx, (x, y) in enumerate(t_loader):
    #     cos = reweight(x, y, dict_prototype)
    #     print(idx, cos.shape)
    #     print(idx, torch.sum(cos, 0).shape)
    #     print(idx, torch.sum(cos, -1).shape)
    #     break
    # for i in range(1, num_category + 1):
    #     plt.subplot(5, 2, i)
    #     plt.tight_layout()
    #     plt.imshow(example_data[i-1], interpolation='none')
    #     plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.show()
    # imshow(torchvision.utils.make_grid(example_data, nrow=5, padding=2))
    # cos = reweight(t_data, t_label, dict_prototype)
    #
    # print(cos)
    # #
    # label_weight = torch.cat((label.float().view(-1, 1), cos.view(-1, 1)), 1)
    # print(label_weight.shape)
    #
    # train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(t_data, label_weight),
    #                             batch_size=100, shuffle=True)
    # for idx, (x, y) in enumerate(train_loader):
    #     label = y[:, 0].long()
    #     weight= y[:, 1].view(-1, 1)
    #
    #     cp_ = torch.ones(x.shape[0], num_category)
    #     weight = cp_ * weight
    #     print(x.shape, label.shape, weight.shape)