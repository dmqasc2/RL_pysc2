from absl import app
from absl import flags
import sys
import torch
from utils import arglist
from runs.minigame import MiniGame
from agent.ddpg import DDPGAgent
from networks.acnetwork_seperated import ActorNet, CriticNet
from utils.memory import Memory
from utils.preprocess import Preprocess

torch.set_default_tensor_type('torch.FloatTensor')
torch.manual_seed(arglist.SEED)

FLAGS = flags.FLAGS
FLAGS(sys.argv)
flags.DEFINE_bool("render", False, "Whether to render with pygame.")

env_names = ["DefeatZerglingsAndBanelings", "DefeatRoaches",
             "CollectMineralShards", "MoveToBeacon", "FindAndDefeatZerglings",
             "BuildMarines", "CollectMineralsAndGas"]


def main(_):
    for map_name in env_names:
        actor = ActorNet()
        critic = CriticNet()
        memory = Memory(limit=1e2,
                        action_shape={'categorical': (arglist.NUM_ACTIONS,),
                                      'screen1': (arglist.FEAT2DSIZE, arglist.FEAT2DSIZE),
                                      'screen2': (arglist.FEAT2DSIZE, arglist.FEAT2DSIZE)},
                        observation_shape={'minimap': (arglist.FEAT2DSIZE, arglist.FEAT2DSIZE),
                                           'screen': (arglist.FEAT2DSIZE, arglist.FEAT2DSIZE),
                                           'nonspatial': (arglist.NUM_ACTIONS,)})
        learner = DDPGAgent(actor, critic, memory)
        preprocess = Preprocess()
        game = MiniGame(map_name, learner, preprocess, nb_episodes=10000)
        game.run()
    return 0


if __name__ == '__main__':
    app.run(main)
