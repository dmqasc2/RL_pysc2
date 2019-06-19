import numpy as np
from utils import arglist
from pysc2.lib import features


class Preprocess:
    def __init__(self):
        self.num_screen_channels = len(features.SCREEN_FEATURES)
        self.num_minimap_channels = len(features.MINIMAP_FEATURES)
        self.num_flat_obs = arglist.NUM_ACTIONS
        self.available_actions_channels = arglist.NUM_ACTIONS

    def get_observation(self, state):
        obs_flat = state.observation['available_actions']
        obs_flat = self._onehot1d(obs_flat)

        obs = {'minimap': state.observation['feature_minimap'],
               'screen': state.observation['feature_screen'],
               'nonspatial': obs_flat}
        return obs

    def preprocess_action(self, act):
        return act

    def postprocess_action(self, act):
        '''
        input: action <FunctionCall>
        output: action <dict of np.array>
        '''
        act_categorical = np.zeros(shape=(arglist.NUM_ACTIONS,), dtype='float32')
        act_categorical[act.function] = 1.

        act_screens = [np.zeros(shape=(1, arglist.FEAT2DSIZE, arglist.FEAT2DSIZE), dtype='float32')] * 2
        i = 0
        for arg in act.arguments:
            if arg != [0]:
                act_screens[i][0, arg[0], arg[1]] = 1.
                i += 1

        act = {'categorical': act_categorical,
               'screen1': act_screens[0],
               'screen2': act_screens[1]}

        return act

    def _onehot1d(self, x):
        y = np.zeros((self.num_flat_obs,), dtype='float32')
        y[x] = 1.
        return y
