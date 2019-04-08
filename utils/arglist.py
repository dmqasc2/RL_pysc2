import numpy
import torch
from pysc2.lib import actions

DEVICE = torch.device('cuda:0')
SEED = 1234
FEAT2DSIZE = 32
NUM_ACTIONS = len(actions.FUNCTIONS)

EPS = np.finfo(np.float32).eps.item()
GAMMA = 0.99
TAU = 0.001
LEARNINGRATE = 0.001