import numpy as np
import torch
from pysc2.lib import actions

DEVICE = torch.device('cuda:0')
SEED = 1234
FEAT2DSIZE = 64
NUM_ACTIONS = len(actions.FUNCTIONS)

EPS = np.finfo(np.float32).eps.item()
GAMMA = 0.99
TAU = 0.001
LEARNINGRATE = 0.001

BatchSize = 5

memory_limit = 1e2,
action_shape = {'categorical': (NUM_ACTIONS,),
                'screen1': (1, FEAT2DSIZE, FEAT2DSIZE),
                'screen2': (1, FEAT2DSIZE, FEAT2DSIZE)},
observation_shape = {'minimap': (7, FEAT2DSIZE, FEAT2DSIZE),
                     'screen': (17, FEAT2DSIZE, FEAT2DSIZE),
                     'nonspatial': (NUM_ACTIONS,)}


