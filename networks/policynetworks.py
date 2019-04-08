'''
<reference>
https://github.com/wing3s/pysc2-rl-mini/blob/master/rl/model.py
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from torch.nn.utils import weight_norm
from torch.autograd import Variable

from utils.arglist import NUM_ACTIONS, SIZE


def init_weights(model):
    if type(model) in [nn.Linear, nn.Conv2d]:
        init.xavier_uniform_(model.weight)
        init.constant_(model.bias, 0)
    elif type(model) in [nn.LSTMCell]:
        init.constant_(model.bias_ih, 0)
        init.constant_(model.bias_hh, 0)


class PolicyNetwork(nn.Module):
    def __init__(self, minimap_channels=17, screen_channels=7, num_action=NUM_ACTIONS):
        super(PolicyNetwork, self).__init__()
        self.eps = None
        self.optimizer = None
        self.num_action = num_action

        # apply paddinga as 'same', padding = (kernel - 1)/2
        self.mconv1 = nn.Conv2d(minimap_channels, 16, 5, stride=1, padding=2)  # shape (N, 16, m, m)
        self.mconv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)  # shape (N, 32, m, m)
        self.sconv1 = nn.Conv2d(screen_channels, 16, 5, stride=1, padding=2)  # shape (N, 16, s, s)
        self.sconv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)  # shape (N, 32, s, s)

        # spatial actor
        state_channels = 32 * 2  # stacking minimap, screen, info
        self.sa_conv3 = nn.Conv2d(state_channels, 1, 1, stride=1)  # shape (N, 65, s, s)

        # non spatial feature
        self.ns_fc3 = nn.Linear(SIZE * SIZE * state_channels, 256)
        # non spatial actor
        self.nsa_fc4 = nn.Linear(256, self.num_action)

        self.apply(init_weights)

    def forward(self, minimap_vb, screen_vb, valid_action_vb):
        """
        Args:
            minimap_vb, (N, # of channel, width, height)
            screen_vb, (N, # of channel, width, height)
            info_vb, (len(info))
            valid_action_vb, (len(observation['available_actions])) same as (num_actions)
        Returns:
            value_vb, (1, 1)
            spatial_policy_vb, (1, s*s)
            non_spatial_policy_vb, (1, num_actions)
            lstm_hidden variables
        """
        x_m = F.relu(self.mconv1(minimap_vb))
        x_m = F.relu(self.mconv2(x_m))
        x_s = F.relu(self.sconv1(screen_vb))
        x_s = F.relu(self.sconv2(x_s))

        x_state = torch.cat((x_m, x_s), dim=1)  # concat along channel dimension

        x_spatial = self.sa_conv3(x_state)
        x_spatial = x_spatial.view(x_spatial.shape[0], -1)
        spatial_policy_vb = F.softmax(x_spatial, dim=1)

        x_non_spatial = x_state.view(x_state.shape[0], -1)
        x_non_spatial = F.relu(self.ns_fc3(x_non_spatial))

        x_non_spatial_policy = self.nsa_fc4(x_non_spatial)
        non_spatial_policy_vb = F.softmax(x_non_spatial_policy, dim=1)
        non_spatial_policy_vb = self._mask_unavailable_actions(non_spatial_policy_vb,
                                                               valid_action_vb)

        return spatial_policy_vb, non_spatial_policy_vb

    def _mask_unavailable_actions(self, policy_vb, valid_action_vb):
        """
        Args:
            policy_vb, (1, num_actions)
            valid_action_vb, (num_actions)
        Returns:
            masked_policy_vb, (1, num_actions)
        """
        masked_policy_vb = policy_vb * valid_action_vb
        masked_policy_vb /= masked_policy_vb.sum(1)
        return masked_policy_vb