import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
# import matplotlib.pyplot as plt
import numpy as np
from model import LeNet_center, ResNet8, ResNet20, ResNet32, WideResNet, NASNetAMobile, LeNet
from DLN_LSTM import MetaOptimizer, LSTMOptimizer
import functools
import higher
import random
from copy import deepcopy
import torch.nn.functional as F
import json
import torch.nn as nn
import time
import argparse
import os
from utils import get_flat_parameters, set_parameters, getPrototype, reweight, DataLoaderX, AverageMeter, accuracy
from torchvision.datasets import ImageNet
from pytorch_metric_learning import distances, losses, miners, reducers, testers
# plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='PyTorch Radiomics Training')
parser.add_argument('--lr_cls_base', default=1e-1, type=float, help='learning rate of classifier')
parser.add_argument('--lr_DLN', default=1e-3, type=float, help='learning rate of classifier')
parser.add_argument('--train_batchsize', default=25, type=int, help='batch size of training data')
parser.add_argument('--val_batchsize', default=100, type=int, help='batch size of validation data')
parser.add_argument('--outer_loop', default=10, type=int, help='outer loop')
parser.add_argument('--inner_loop', default=15, type=int, help='inner loop')
parser.add_argument('--network', '-r', default='ResNet8', help='classifier:ResNet8, ResNet20, ResNet32')
parser.add_argument('--dataset', default='cifar10', help='dataset: cifar10, cifar100')
args = parser.parse_args()

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

test_device = 'cuda:0'

num_train, num_test = 25000, 25000 # 30000, 30000 #

batch_num_META = num_test // args.val_batchsize

reset_flag = False
lr_cls = args.lr_cls_base

def to_(x):
    return x.to(device)

def detach_var(v):
    var = to_(Variable(v.data, requires_grad=True))
    var.retain_grad()
    return var

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def seed_torch(seed=12):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def get_DataLoader(dataset='mnist'):
    print(args.dataset)
    if dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
        test_ = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
        train_data, val_data = torch.utils.data.random_split(trainset, [num_train, num_test])
        num_category = 10

    elif dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
             ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform_train)
        test_ = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
        train_data, val_data = torch.utils.data.random_split(trainset, [num_train, num_test])
        num_category = 100

    return train_data, val_data, test_, num_category

def get_Model(num_cate):
    'ResNet8, ResoNet20, ResNet32, WideResNet'
    print(args.network)
    if args.network == 'ResNet8':
        net = ResNet8(num_cate)
        
    elif args.network == 'ResNet20':
       net = ResNet20(num_classes=num_cate)
       
    elif args.network == 'ResNet32':
        net = ResNet32(num_classes=num_cate)

    return net

seed_torch()

DLN = MetaOptimizer(dvs=device)

LSTM_Meta= LSTMOptimizer().to(device)
print(LSTM_Meta)
optimizer= optim.Adam(LSTM_Meta.parameters(), lr=args.lr_DLN)

CE = nn.CrossEntropyLoss()
_, _, test_, num_category = get_DataLoader(args.dataset)
test_loader = torch.utils.data.DataLoader(test_, batch_size=args.val_batchsize, shuffle=True, num_workers=0, pin_memory=False)

net = get_Model(num_category)
net = net.to(device)

inner_opt = optim.SGD(net.parameters(), lr=args.lr_cls_base)

copy_model = deepcopy(net)

net.train()
DLN.train()
LSTM_Meta.train()

record_loss = []
record_acc  = []
record_acc_test=[]

n_params = 0
for name, p in DLN.all_named_parameters():
    n_params += int(np.prod(p.size()))

hidden_states = [
    to_(Variable(torch.zeros(n_params, LSTM_Meta.hidden_sz))),
    to_(Variable(torch.zeros(n_params, LSTM_Meta.hidden_sz))),
    to_(Variable(torch.zeros(n_params, LSTM_Meta.hidden_sz))),
    to_(Variable(torch.zeros(n_params, LSTM_Meta.output_sz)))
]
cell_states   = [
    to_(Variable(torch.zeros(n_params, LSTM_Meta.hidden_sz))),
    to_(Variable(torch.zeros(n_params, LSTM_Meta.hidden_sz))),
    to_(Variable(torch.zeros(n_params, LSTM_Meta.hidden_sz))),
    to_(Variable(torch.zeros(n_params, LSTM_Meta.output_sz)))
]

LSTM_Loss = 0
step = 0
for out in range(args.outer_loop):
    train_data, val_data, test_, num_category = get_DataLoader(args.dataset)

    t_loader = DataLoaderX(train_data, batch_size=args.train_batchsize, shuffle=True, num_workers=0, pin_memory=False)
    v_loader = DataLoaderX(val_data, batch_size=args.val_batchsize, shuffle=True, num_workers=0, pin_memory=False)

    triter = iter(t_loader)

    for btx_id, (v_x, v_y) in enumerate(v_loader):
        start = time.perf_counter()
        net.train()
        v_x, v_y = v_x.to(device), v_y.to(device)

        with torch.no_grad():
            trainset = []
            for i in range(args.inner_loop):
                d_tuple = next(triter, True)
                if d_tuple == True:
                    triter = iter(t_loader)
                    continue
                x, y = d_tuple
                trainset.append((x, y))

        with higher.innerloop_ctx(net, inner_opt) as (fnet, in_opt):
            for idx, (x, y) in enumerate(trainset):
                x, y = x.to(device), y.to(device)

                output = fnet(x)
                D_Loss = torch.mean(DLN(output, y))
                in_opt.step(D_Loss)

            pre_vl  = fnet(v_x)
            single_loss = CE(pre_vl, v_y.long())

            offset = 0
            result_params = {}
            hidden_states2 = [
                to_(Variable(torch.zeros(n_params, LSTM_Meta.hidden_sz))),
                to_(Variable(torch.zeros(n_params, LSTM_Meta.hidden_sz))),
                to_(Variable(torch.zeros(n_params, LSTM_Meta.hidden_sz))),
                to_(Variable(torch.zeros(n_params, LSTM_Meta.output_sz)))
            ]
            cell_states2 = [
                to_(Variable(torch.zeros(n_params, LSTM_Meta.hidden_sz))),
                to_(Variable(torch.zeros(n_params, LSTM_Meta.hidden_sz))),
                to_(Variable(torch.zeros(n_params, LSTM_Meta.hidden_sz))),
                to_(Variable(torch.zeros(n_params, LSTM_Meta.output_sz)))
            ]
            dict_p = {}
            for name, p in DLN.all_named_parameters():
                cur_sz = int(np.prod(p.size()))
                gradients = torch.autograd.grad(single_loss, p, retain_graph=True)[0]
                gradients = detach_var(gradients.view(cur_sz, 1))  # * 1e2
                updates, new_hidden, new_cell = LSTM_Meta(
                    gradients,
                    [h[offset:offset + cur_sz] for h in hidden_states],
                    [c[offset:offset + cur_sz] for c in cell_states]
                )
                for i in range(len(new_hidden)):
                    hidden_states2[i][offset:offset + cur_sz] = new_hidden[i]
                    cell_states2[i][offset:offset + cur_sz] = new_cell[i]
                result_params[name] = p - updates.view(*p.size()) * 1e-3
                result_params[name].retain_grad()
                offset += cur_sz
            hidden_states = hidden_states2
            cell_states = cell_states2
            for name, p in DLN.all_named_parameters():
                rsetattr(DLN, name, result_params[name])
            assert len(list(DLN.all_named_parameters()))

        with higher.innerloop_ctx(net, inner_opt) as (fnet, in_opt):
            for idx, (x, y) in enumerate(trainset):
                x, y = x.to(device), y.to(device)
                output = fnet(x)
                D_Loss = torch.mean(DLN(output, y))
                in_opt.step(D_Loss)

            pre_vl = fnet(v_x)
            total_loss = CE(pre_vl, v_y.long())
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()

        new_DLN = to_(MetaOptimizer(dvs=device))
        new_DLN.load_state_dict(DLN.state_dict())
        DLN = new_DLN
        hidden_states = [detach_var(v) for v in hidden_states]
        cell_states = [detach_var(v) for v in cell_states]
        total_loss.detach_()

        torch.cuda.empty_cache()

        copy_model.load_state_dict(fnet.state_dict())
        param = get_flat_parameters(copy_model).data
        set_parameters(net, param)
        inner_opt = optim.SGD(net.parameters(), lr=lr_cls)

        ne = deepcopy(net)
        ne = ne.to(test_device)
        ne.load_state_dict(net.state_dict())
        ne.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_id, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(test_device), targets.to(test_device)
                outputs = ne(inputs)
                total += targets.size(0)
                _, predicted = outputs.max(1)  # judge max elements in predicted`s Row(1:Row     0:Column)
                correct += predicted.eq(targets).float().sum().item()
        end = time.perf_counter()
        print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f \t\t time:%.4f'
                         % (out, args.outer_loop, btx_id,
                            len(v_loader) + 1, total_loss.item(), (100. * correct / total), (end-start)))
