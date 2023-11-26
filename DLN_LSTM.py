import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from meta_module import *
import numpy as np
from torch.autograd import Variable

class MetaOptimizer(MetaModule):

    def __init__(self, n_layers=5, layer_size=40, dvs='cuda:0'):
        super(MetaOptimizer, self).__init__()
        self.bias = False

        lst = [4, 40, 40, 40, 40, 40] #32, 64, 128, 256, 512

        inp_size = 4
        self.layers = {}
        for i in range(0, n_layers-1):
            self.layers[f'mat_{i}'] = MetaLinear(lst[i], lst[i+1], self.bias, dvs=dvs)
            inp_size = layer_size

        self.layers['final_mat'] = MetaLinear(lst[-1], 1, self.bias, dvs)
        self.layers = nn.ModuleDict(self.layers)
        self.activation = nn.LeakyReLU()
        # self.out_active = nn.Softplus()

        self.init_weight()

    def all_named_parameters(self):
        return [(k, v) for k, v in self.named_parameters()]

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='leakay_relu')

    def forward(self, x, y):
        targets_onehot = torch.zeros_like(x)
        targets_onehot.zero_()
        targets_onehot.scatter_(1, y.long().unsqueeze(-1), 1).float()
        y_onehot = targets_onehot
        bs, num_cate = x.size()

        x = torch.sigmoid(x)
        truth_pred = torch.sum(x * y_onehot, 1).view(-1, 1, 1)
        false_pred = x * (1 - y_onehot)
        false_pred = false_pred.view(-1, num_cate, 1)
        cp_true_pred = torch.ones_like(false_pred) * truth_pred
        predict = torch.cat((cp_true_pred, false_pred), -1)
        predict = predict.view(-1, 2)

        false_label = (1 - y_onehot).view(-1, 1)
        predict = predict * false_label

        label = torch.zeros(predict.shape[0], 2)
        label = (label + torch.tensor([1., 0.]))
        y_label = (label).cuda(x.device)  # .unsqueeze(1)
        y_label = y_label * false_label

        x = torch.cat((predict, y_label), -1)
        # x = torch.cat((x, y_onehot), -1)
        x = x.view(-1, 4)
        cur_layer = 0
        while f'mat_{cur_layer}' in self.layers:
            x = self.activation(self.layers[f'mat_{cur_layer}'](x))
            cur_layer += 1

        MLN_loss = self.layers['final_mat'](x)# self.out_active()

        return MLN_loss

class LSTMOptimizer(nn.Module):
    def __init__(self, preproc=True, hidden_sz=64, preproc_factor=10.0):
        super(LSTMOptimizer, self).__init__()
        self.hidden_sz = hidden_sz
        self.output_sz = 1
        if preproc:
            self.recurs = nn.LSTMCell(2, self.hidden_sz)
        else:
            self.recurs = nn.LSTMCell(1, hidden_sz)

        self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz)
        self.recurs3 = nn.LSTMCell(hidden_sz, hidden_sz)
        self.recurs4 = nn.LSTMCell(hidden_sz, self.output_sz)

        self.preproc = preproc
        self.preproc_factor = preproc_factor
        self.preproc_threshold = np.exp(-preproc_factor)

        # self.init_weight()

    def init_weight(self):
        for name, param in self.recurs.named_parameters():
            if name.startswith("weight"):
                nn.init.orthogonal_(param.data)
            elif name.startswith("bias"):
                nn.init.zeros_(param.data)
        for name, param in self.recurs2.named_parameters():
            if name.startswith("weight"):
                nn.init.orthogonal_(param.data)
            elif name.startswith("bias"):
                nn.init.zeros_(param.data)
        for name, param in self.recurs3.named_parameters():
            if name.startswith("weight"):
                nn.init.orthogonal_(param.data)
            elif name.startswith("bias"):
                nn.init.zeros_(param.data)
        for name, param in self.recurs4.named_parameters():
            if name.startswith("weight"):
                nn.init.orthogonal_(param.data)
            elif name.startswith("bias"):
                nn.init.zeros_(param.data)

    def forward(self, inp, hidden, cell): #
        if self.preproc:
            param_inp2 = torch.zeros(inp.size()[0], 2).to(inp.device)
            keep_grads = (torch.abs(inp) >= self.preproc_threshold).squeeze()
            param_inp2[:, 0][keep_grads] = (torch.log(torch.abs(inp[keep_grads]) + 1e-8) / self.preproc_factor).squeeze()
            param_inp2[:, 1][keep_grads] = torch.sign(inp[keep_grads]).squeeze()

            param_inp2[:, 0][~keep_grads] = -1
            param_inp2[:, 1][~keep_grads] = (float(np.exp(self.preproc_factor)) * inp[~keep_grads]).squeeze()
            inp = Variable(param_inp2)
        hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
        hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
        hidden2, cell2 = self.recurs3(hidden1, (hidden[2], cell[2]))
        hidden3, cell3 = self.recurs4(hidden2, (hidden[3], cell[3]))
        return hidden3, [hidden0, hidden1, hidden2, hidden3], [cell0, cell1, cell2, cell3]

if __name__ == '__main__':
    class Teacher(nn.Module):
        def __init__(self):
            super(Teacher, self).__init__()

            self.input_lyr = nn.Sequential(
                nn.Linear(10, 128),  # 87440
                nn.ReLU(),
                nn.Linear(128, 128),  # 87440
                nn.ReLU(),
                nn.Linear(128, 128),  # 87440
                nn.ReLU(),
                nn.Linear(128, 128),  # 87440
                nn.ReLU(),
                nn.Linear(128, 128),  # 87440
                nn.ReLU(),
                nn.Linear(128, 128),  # 87440
                nn.ReLU(),
                nn.Linear(128, 128),  # 87440
                nn.ReLU(),
                nn.Linear(128, 128),  # 87440
                nn.ReLU(),
                nn.Linear(128, 128),  # 87440
                nn.ReLU(),
                nn.Linear(128, 4),  # 86815
                nn.Softmax()
            )

        def forward(self, x):
            return self.input_lyr(x)

    def count_param(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    model = Teacher()
    print(f'Total number of parameters: {count_param(model)}')