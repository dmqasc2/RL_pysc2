'''
<reference>
https://github.com/wing3s/pysc2-rl-mini/blob/master/rl/model.py

 - LSTM policy is NOT implemented
 - unified Actor Critic
'''

import math
from math import floor

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from torch.nn.utils import weight_norm
from torch.autograd import Variable

from pysc2.lib import actions
from pysc2.lib import features

import numpy as np


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def init_weights(model):
    if type(model) in [nn.Linear, nn.Conv2d]:
        init.xavier_uniform_(model.weight)
        init.constant_(model.bias, 0)
    elif type(model) in [nn.LSTMCell]:
        init.constant_(model.bias_ih, 0)
        init.constant_(model.bias_hh, 0)


def make_one_hot_1d(labels, dtype, C=2):
    '''
    Reference: https://lirnli.wordpress.com/2017/09/03/one-hot-encoding-in-pytorch/
    Parameters
    ----------
    labels : N, where N is batch size.
    dtype: Cuda or not
    C : number of classes in labels.

    Returns
    -------
    target : N x C
    '''
    out = Variable(dtype(labels.size(0), C).zero_())
    index = labels.contiguous().view(-1, 1).long()
    return out.scatter_(1, index, 1)


class AtariNet(torch.nn.Module):
    def __init__(self,
                 minimap_channels,
                 screen_channels,
                 screen_resolution,
                 nonspatial_obs_dim,
                 num_action):
        super(AtariNet, self).__init__()

        # spatial features
        # apply paddinga as 'same', padding = (kernel - 1)/2
        self.minimap_conv_layers = nn.Sequential(
            nn.Conv2d(minimap_channels, 16, 8, stride=4),  # shape (N, 16, m, m)
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),  # shape (N, 32, m, m)
            nn.ReLU(),
            Flatten()
        )

        self.screen_conv_layers = nn.Sequential(
            nn.Conv2d(screen_channels, 16, 8, stride=4),  # shape (N, 16, m, m)
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),  # shape (N, 32, m, m)
            nn.ReLU(),
            Flatten()
        )

        # non-spatial features
        self.nonspatial_dense = nn.Sequential(
            nn.Linear(nonspatial_obs_dim, 256),
            nn.Tanh()
        )

        # calculated conv. output shape for input resolutions
        shape_conv = self._conv_output_shape(screen_resolution, kernel_size=8, stride=4)
        shape_conv = self._conv_output_shape(shape_conv, kernel_size=4, stride=2)

        # state representations
        self.layer_hidden = nn.Sequential(nn.Linear(32 * shape_conv[0] * shape_conv[1] + 256, 256),
                                          nn.ReLU()
                                         )
        # output layers
        self.layer_value = nn.Linear(256, 1)
        self.layer_action = nn.Linear(256, num_action)
        self.layer_screen1_x = nn.Linear(256, screen_resolution[0])
        self.layer_screen1_y = nn.Linear(256, screen_resolution[1])
        self.layer_screen2_x = nn.Linear(256, screen_resolution[0])
        self.layer_screen2_y = nn.Linear(256, screen_resolution[1])

        self.apply(init_weights)  # weight initialization
        self.train()  # train mode

    def forward(self, obs_minimap, obs_screen, obs_nonspatial, valid_actions):
        # process observations
        m = self.minimap_conv_layers(obs_minimap)
        s = self.screen_conv_layers(obs_screen)
        n = self.nonspatial_dense(obs_nonspatial)

        m = m.view(m.size(0), -1)
        s = s.view(s.size(0), -1)
        state_representation = self.layer_hidden(torch.cat([m, s, n]))
        v = self.layer_value(state_representation)
        pol_categorical = self.layer_action(state_representation)
        pol_categorical = self._mask_unavailable_actions(pol_categorical)
        pol_screen1_x = self.layer_screen1_x(state_representation)
        pol_screen1_y = self.layer_screen1_y(state_representation)
        pol_screen2_x = self.layer_screen2_x(state_representation)
        pol_screen2_y = self.layer_screen2_y(state_representation)

        return v, pol_categorical, pol_screen1_x, pol_screen1_y, pol_screen2_x, pol_screen2_y

    def _conv_output_shape(self, h_w, kernel_size=1, stride=1, pad=0, dilation=1):
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)
        h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
        w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
        return h, w

    def _mask_unavailable_actions(self, policy, valid_actions):
        """
            Args:
                policy_vb, (1, num_actions)
                valid_action_vb, (num_actions)
            Returns:
                masked_policy_vb, (1, num_actions)
        """
        masked_policy_vb = policy * valid_actions
        masked_policy_vb /= masked_policy_vb.sum(1)
        return masked_policy_vb


class FullyConvNet(torch.nn.Module):
    def __init__(self,
                 minimap_channels,
                 screen_channels,
                 screen_resolution,
                 nonspatial_obs_dim,
                 num_action):
        super(FullyConvNet, self).__init__()

        # spatial features
        # apply paddinga as 'same', padding = (kernel - 1)/2
        self.minimap_conv_layers = nn.Sequential(
            nn.Conv2d(minimap_channels, 16, 5, stride=1, padding=2),  # shape (N, 16, m, m)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),  # shape (N, 32, m, m)
            nn.ReLU()
        )

        self.screen_conv_layers = nn.Sequential(
            nn.Conv2d(screen_channels, 16, 5, stride=1, padding=2),  # shape (N, 16, m, m)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),  # shape (N, 32, m, m)
            nn.ReLU()
        )

        # non-spatial features
        self.nonspatial_dense = nn.Sequential(
            nn.Linear(nonspatial_obs_dim, 256),
            nn.Tanh()
        )

        # calculated conv. output shape for input resolutions
        shape_conv = self._conv_output_shape(screen_resolution, kernel_size=8, stride=4)
        shape_conv = self._conv_output_shape(shape_conv, kernel_size=4, stride=2)

        # state representations
        self.layer_hidden = nn.Sequential(nn.Linear(2 * 32 * shape_conv[0] * shape_conv[1], 256),
                                          nn.ReLU()
                                          )
        # output layers
        self.layer_value = nn.Linear(256, 1)
        self.layer_action = nn.Linear(256, num_action)
        self.layer_screen1 = nn.Conv2d(64, 1, 1)
        self.layer_screen2 = nn.Conv2d(64, 1, 1)

        self.apply(init_weights)  # weight initialization
        self.train()  # train mode

    def forward(self, obs_minimap, obs_screen, obs_nonspatial, valid_actions):
        # process observations
        m = self.minimap_conv_layers(obs_minimap)
        s = self.screen_conv_layers(obs_screen)
        # n = self.nonspatial_dense(obs_nonspatial)

        state_representation = torch.cat([m, s], dim=1)
        state_representation_dense = self.layer_hidden(state_representation)
        v = self.layer_value(state_representation_dense)
        pol_categorical = self.layer_action(state_representation_dense)
        pol_categorical = self._mask_unavailable_actions(pol_categorical)

        # conv. output
        pol_screen1 = self.layer_screen1_x(state_representation)
        pol_screen2 = self.layer_screen2_x(state_representation)

        return v, pol_categorical, pol_screen1, pol_screen2

    def _conv_output_shape(self, h_w, kernel_size=1, stride=1, pad=0, dilation=1):
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)
        h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
        w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
        return h, w

    def _mask_unavailable_actions(self, policy, valid_actions):
        """
            Args:
                policy_vb, (1, num_actions)
                valid_action_vb, (num_actions)
            Returns:
                masked_policy_vb, (1, num_actions)
        """
        masked_policy_vb = policy * valid_actions
        masked_policy_vb /= masked_policy_vb.sum(1)
        return masked_policy_vb




