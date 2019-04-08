from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.env import sc2_env
from pysc2.lib import actions as Actions

from absl import app
from absl import flags
import sys

import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from torch.autograd import Variable
from torch.distributions import Categorical

DEVICE = torch.device('cuda:0')
torch.set_default_tensor_type('torch.FloatTensor')
SEED = 1234
SIZE = 32
torch.manual_seed(SEED)

nb_episodes = 10000
nb_max_steps = 2000

FLAGS = flags.FLAGS
FLAGS(sys.argv)
flags.DEFINE_bool("render", False, "Whether to render with pygame.")

agent_format = sc2_env.AgentInterfaceFormat(
    feature_dimensions=sc2_env.Dimensions(
        screen=(SIZE, SIZE),
        minimap=(SIZE, SIZE), )
)

env_names = ["DefeatZerglingsAndBanelings", "DefeatRoaches",
             "CollectMineralShards", "MoveToBeacon", "FindAndDefeatZerglings",
             "BuildMarines", "CollectMineralsAndGas"]


def init_weights(model):
    if type(model) in [nn.Linear, nn.Conv2d]:
        init.xavier_uniform_(model.weight)
        init.constant_(model.bias, 0)
    elif type(model) in [nn.LSTMCell]:
        init.constant_(model.bias_ih, 0)
        init.constant_(model.bias_hh, 0)


def preprocess_available_actions(available_actions, max_action=len(Actions.FUNCTIONS)):
    a_actions = np.zeros((max_action), dtype='float32')
    a_actions[available_actions] = 1.
    return a_actions


def post_preprocessing_action(actions):
    """
    Transform selected non_spatial and spatial actions into pysc2 FunctionCall
    Args:
        non_spatial_action: ndarray, shape (1, 1)
        spatial_action: ndarray, shape (1, 1)
    Returns:
        FunctionCall as action for pysc2_env
    """
    act_id = actions[1]
    target = actions[0]
    target_point = [
        int(target % SIZE),
        int(target // SIZE)
    ]  # (x, y)

    act_args = []
    for arg in Actions.FUNCTIONS[act_id].args:
        if arg.name in ('screen', 'minimap', 'screen2'):
            act_args.append(target_point)
        else:
            act_args.append([0])
    return Actions.FunctionCall(act_id, act_args)


class SimplePolicy(nn.Module):
    def __init__(self, minimap_channels=17, screen_channels=7, num_action=len(Actions.FUNCTIONS)):
        super(SimplePolicy, self).__init__()
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
        self.train()

        self.saved_log_probs_spatial = []
        self.saved_log_probs_nonspatial = []
        self.rewards = []

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

    def select_action(self, minimap_vb, screen_vb, valid_action_vb):
        valid_action_vb = preprocess_available_actions(valid_action_vb, max_action=len(Actions.FUNCTIONS))

        minimap_vb = minimap_vb.astype('float32')
        screen_vb = screen_vb.astype('float32')

        minimap_vb = torch.from_numpy(np.expand_dims(minimap_vb, 0)).to(DEVICE)
        screen_vb = torch.from_numpy(np.expand_dims(screen_vb, 0)).to(DEVICE)
        valid_action_vb = torch.from_numpy(np.expand_dims(valid_action_vb, 0)).to(DEVICE)

        probs_spatial, probs_nonspatial = self.forward(minimap_vb, screen_vb, valid_action_vb)
        # spatial action
        action_spatial = torch.argmax(probs_spatial, dim=-1)
        self.saved_log_probs_spatial.append(probs_spatial[0][action_spatial.item()].log())

        # m_spatial = Categorical(probs_spatial)
        # action_spatial = m_spatial.sample()
        # self.saved_log_probs_spatial.append(m_spatial.log_prob(action_spatial))

        # non-spatial action
        action_nonspatial = torch.argmax(probs_nonspatial, dim=-1)
        self.saved_log_probs_nonspatial.append(probs_nonspatial[0][action_nonspatial.item()].log())

        # m_nonspatial = Categorical(probs_nonspatial)
        # action_nonspatial = m_nonspatial.sample()
        # self.saved_log_probs_nonspatial.append(m_nonspatial.log_prob(action_nonspatial))

        try:
            return action_spatial.item(), action_nonspatial.item()
        except RuntimeError:
            print('error')

    def finish_episode(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob_spatial, log_prob_nonspatial, R in zip(self.saved_log_probs_spatial, self.saved_log_probs_nonspatial, returns):
            policy_loss.append((-log_prob_spatial * R) + (-log_prob_nonspatial * R))
        self.optimizer.zero_grad()
        policy_loss = torch.tensor(policy_loss, requires_grad=True).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs_nonspatial[:]
        del self.saved_log_probs_spatial[:]


def run(env_name):
    env = sc2_env.SC2Env(
        map_name=env_name,  # "BuildMarines",
        step_mul=16,
        visualize=False,
        agent_interface_format=[agent_format])

    policy = SimplePolicy()
    policy.to(DEVICE)
    policy.optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    policy.eps = np.finfo(np.float32).eps.item()

    reward_cumulative = []
    for i_episode in range(nb_episodes):
        state = env.reset()[0]
        for t in range(1, nb_max_steps):  # Don't infinite loop while learning
            actions = policy.select_action(state.observation['feature_screen'],
                                           state.observation['feature_minimap'],
                                           state.observation['available_actions'])
            actions = post_preprocessing_action(actions)
            # print(actions)
            state = env.step(actions=[actions])[0]
            policy.rewards.append(state.reward)
            if state.last():
                cum_reward = state.observation["score_cumulative"]
                reward_cumulative.append(cum_reward[0])
                break
        policy.finish_episode()

    env.close()
    print(reward_cumulative)
    print(np.mean(reward_cumulative))


def main(_):
    for e in env_names:
        run(e)
    return 0


if __name__ == '__main__':
    app.run(main)