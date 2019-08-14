import torch
import numpy as np
from pysc2.lib import actions
from torch.nn.functional import gumbel_softmax
from torch.distributions import Categorical
from utils import arglist


class Agent(object):
    def preprocess_available_actions(self, available_actions, max_action=arglist.NUM_ACTIONS):
        a_actions = np.zeros(max_action, dtype='float32')
        a_actions[available_actions] = 1.
        return a_actions

    def _mask_unavailable_actions(self, policy, valid_actions):
        """
            Args:
                policy_vb, (1, num_actions)
                valid_action_vb, (num_actions)
            Returns:
                masked_policy_vb, (1, num_actions)
        """
        valid_actions = torch.from_numpy(valid_actions).to(arglist.DEVICE)
        masked_policy_vb = policy * valid_actions
        # masked_policy_vb /= masked_policy_vb.sum(1)
        return masked_policy_vb

    def _test_valid_action(self, function_id, valid_actions):
        if valid_actions[function_id] == 1:
            return True
        else:
            return False

    def select_action(self, obs, valid_actions):
        '''
        from logit to pysc2 actions
        :param logits: {'categorical': [], 'screen1': [], 'screen2': []}
        :return: FunctionCall form of action
        '''
        obs_torch = {'categorical': 0, 'screen1': 0, 'screen2': 0}
        for o in obs:
            x = obs[o].astype('float32')
            x = np.expand_dims(x, 0)
            obs_torch[o] = torch.from_numpy(x).to(arglist.DEVICE)

        logits = self.actor(obs_torch)
        prob_categorical = torch.nn.Softmax(dim=-1)(logits['categorical']).detach()
        prob_categorical = self._mask_unavailable_actions(prob_categorical, valid_actions)
        dist = Categorical(prob_categorical)

        function_id = None
        while True:
            try:
                function_id = dist.sample().item()
            except RuntimeError:
                print('.')
            if function_id is not None:
                break

        is_valid_action = self._test_valid_action(function_id, valid_actions)
        while not is_valid_action:
            function_id = dist.sample().item()
            is_valid_action = self._test_valid_action(function_id, valid_actions)

        p = torch.nn.Softmax(dim=-1)(logits['screen1'].view(1, -1)).detach()
        pos_screen1 = Categorical(p).sample().item()
        p = torch.nn.Softmax(dim=-1)(logits['screen2'].view(1, -1)).detach()
        pos_screen2 = Categorical(p).sample().item()

        pos = [[int(pos_screen1 % arglist.FEAT2DSIZE), int(pos_screen1 // arglist.FEAT2DSIZE)],
               [int(pos_screen2 % arglist.FEAT2DSIZE), int(pos_screen2 // arglist.FEAT2DSIZE)]]  # (x, y)

        args = []
        cnt = 0
        for arg in actions.FUNCTIONS[function_id].args:
            if arg.name in ['screen', 'screen2', 'minimap']:
                args.append(pos[cnt])
                cnt += 1
            else:
                args.append([0])

        action = actions.FunctionCall(function_id, args)
        return action

    @DeprecationWarning
    def select_action_old(self, obs, valid_actions):
        '''
        from logit to pysc2 actions
        :param logits: {'categorical': [], 'screen1': [], 'screen2': []}
        :return: FunctionCall form of action
        '''
        obs_torch = {'categorical': 0, 'screen1': 0, 'screen2': 0}
        for o in obs:
            x = obs[o].astype('float32')
            x = np.expand_dims(x, 0)
            obs_torch[o] = torch.from_numpy(x).to(arglist.DEVICE)

        logits = self.actor(obs_torch)
        logits['categorical'] = self._mask_unavailable_actions(logits['categorical'], valid_actions)
        tau = 1.0
        function_id = gumbel_softmax(logits['categorical'], tau=1e-10, hard=True)
        function_id = function_id.argmax().item()
        logits['categorical'].cpu().item()
        # select an action until it is valid.
        is_valid_action = self._test_valid_action(function_id, valid_actions)
        while not is_valid_action:
            tau *= 10
            function_id = gumbel_softmax(logits['categorical'], tau=tau, hard=True)
            function_id = function_id.argmax().item()
            is_valid_action = self._test_valid_action(function_id, valid_actions)

        pos_screen1 = gumbel_softmax(logits['screen1'].view(1, -1), hard=True).argmax().item()
        pos_screen2 = gumbel_softmax(logits['screen2'].view(1, -1), hard=True).argmax().item()

        pos = [[int(pos_screen1 % arglist.FEAT2DSIZE), int(pos_screen1 // arglist.FEAT2DSIZE)],
               [int(pos_screen2 % arglist.FEAT2DSIZE), int(pos_screen2 // arglist.FEAT2DSIZE)]]  # (x, y)

        args = []
        cnt = 0
        for arg in actions.FUNCTIONS[function_id].args:
            if arg.name in ['screen', 'screen2', 'minimap']:
                args.append(pos[cnt])
                cnt += 1
            else:
                args.append([0])

        action = actions.FunctionCall(function_id, args)
        return action
