import torch
from torch import nn
from torch.nn import init
from utils import arglist


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        t, n = x.size(0), x.size(1)
        # merge batch and seq dimensions
        x_reshape = x.contiguous().view(t * n, x.size(2))
        y = self.module(x_reshape)
        # We have to reshape Y
        y = y.contiguous().view(t, n, y.size()[1])
        return y


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Dense2Conv(nn.Module):
    def forward(self, input):
        out = torch.repeat_interleave(input, arglist.FEAT2DSIZE * arglist.FEAT2DSIZE)
        out = out.view(-1, input.shape[1], arglist.FEAT2DSIZE, arglist.FEAT2DSIZE)
        return out


def init_weights(model):
    if type(model) in [nn.Linear, nn.Conv2d]:
        init.xavier_uniform_(model.weight)
        init.constant_(model.bias, 0)
    elif type(model) in [nn.LSTMCell]:
        init.constant_(model.bias_ih, 0)
        init.constant_(model.bias_hh, 0)
