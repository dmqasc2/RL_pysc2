'''
Actor and critic networks share conv. layers to process minimap & screen.
'''

from math import floor
import torch
import torch.nn as nn
from torch.nn import init


minimap_channels = 7
screen_channels = 17

conv_minimap = nn.Sequential(
            nn.Conv2d(minimap_channels, 16, 5, stride=1, padding=2),  # shape (N, 16, m, m)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),  # shape (N, 32, m, m)
            nn.ReLU())

conv_screen = nn.Sequential(
            nn.Conv2d(screen_channels, 16, 5, stride=1, padding=2),  # shape (N, 16, m, m)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),  # shape (N, 32, m, m)
            nn.ReLU())


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


def init_weights(model):
    if type(model) in [nn.Linear, nn.Conv2d]:
        init.xavier_uniform_(model.weight)
        init.constant_(model.bias, 0)
    elif type(model) in [nn.LSTMCell]:
        init.constant_(model.bias_ih, 0)
        init.constant_(model.bias_hh, 0)


class ActorNet(torch.nn.Module):
    def __init__(self,
                 screen_resolution,
                 nonspatial_obs_dim,
                 num_action):
        super(ActorNet, self).__init__()

        # spatial features
        # apply paddinga as 'same', padding = (kernel - 1)/2
        self.minimap_conv_layers = conv_minimap
        self.screen_conv_layers = conv_screen

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

    def forward(self, obs, valid_actions):
        obs_minimap = obs['minimap']
        obs_screen = obs['screen']
        obs_nonspatial = obs['nonspatial']
        
        # process observations
        m = self.minimap_conv_layers(obs_minimap)
        s = self.screen_conv_layers(obs_screen)
        # n = self.nonspatial_dense(obs_nonspatial)

        state_representation = torch.cat([m, s], dim=1)
        state_representation_dense = self.layer_hidden(state_representation)
        v = self.layer_value(state_representation_dense)
        pol_categorical = self.layer_action(state_representation_dense)
        pol_categorical = self._mask_unavailable_actions(pol_categorical, valid_actions)

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


class CriticNet(torch.nn.Module):
    def __init__(self,
                 screen_resolution,
                 nonspatial_obs_dim,
                 num_action):
        super(CriticNet, self).__init__()

        # spatial features
        # apply paddinga as 'same', padding = (kernel - 1)/2
        self.minimap_conv_layers = conv_minimap
        self.screen_conv_layers = conv_screen

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
