import numpy as np
import torch
from pysc2.lib import actions

DEVICE = torch.device('cuda:0')

SEED = 1234
FEAT2DSIZE = 64
NUM_ACTIONS = len(actions.FUNCTIONS)
EPS = np.finfo(np.float32).eps.item()

action_shape = {'categorical': (NUM_ACTIONS,),
                'screen1': (1, FEAT2DSIZE, FEAT2DSIZE),
                'screen2': (1, FEAT2DSIZE, FEAT2DSIZE)}

observation_shape = {'minimap': (7, FEAT2DSIZE, FEAT2DSIZE),
                     'screen': (17, FEAT2DSIZE, FEAT2DSIZE),
                     'nonspatial': (NUM_ACTIONS,)}


# DDPG parameters
class DDPG:
    GAMMA = 0.99
    TAU = 0.001
    LEARNINGRATE = 0.0001
    BatchSize = 128
    memory_limit = int(5e5)


# PPO parameters
class PPO:
    gamma = 0.99
    lamda = 0.98
    hidden = 64
    critic_lr = 0.0003
    actor_lr = 0.0003
    BatchSize = 5
    l2_rate = 0.001
    max_kl = 0.01
    clip_param = 0.2
    memory_limit = int(1e2)
