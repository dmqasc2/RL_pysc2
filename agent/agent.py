from numpy as np
from pysc2.lib import actions
from torch.nn.functional import gumbel_softmax
from utils import arglist



class Agent(object):
    def preprocess_available_actions(self, available_actions, max_action=arglist.NUM_ACTIONS):
        a_actions = np.zeros((max_action), dtype='float32')
        a_actions[available_actions] = 1.
        return a_actions

    def select_action(self, logits):
        '''
        from logit to pysc2 actions
        :param logits: {'categorical': [], 'screen1': [], 'screen2': []}
        :return: FunctionCall form of action
        '''
        function_id = gumbel_softmax(logits['categorical'], hard=True)
        function_id = function_id.argmax().item()

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