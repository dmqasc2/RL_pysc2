from absl import app
from absl import flags
import sys
import torch
from utils import arglist
from run.minigame import MiniGame
from agent.reinforce import Learner
from networks.policynetworks import PolicyNetwork

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
        policy = PolicyNetwork(num_action=arglist.NUM_ACTIONS)
        learner = Learner(policy)
        game = MiniGame(map_name, learner, nb_episodes=10000)
        game.run()
    return 0


if __name__ == '__main__':
    app.run(main)