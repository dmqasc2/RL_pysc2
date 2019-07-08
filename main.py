from absl import app
from absl import flags
import sys
import torch
from utils import arglist
from runs.minigame import MiniGame
from agent.ddpg import DDPGAgent
from agent.ppo import PPOAgent
from networks.acnetwork_seperated import ActorNet, CriticNet
from utils.memory import Memory, EpisodeMemory
from utils.preprocess import Preprocess

torch.set_default_tensor_type('torch.FloatTensor')
torch.manual_seed(arglist.SEED)

FLAGS = flags.FLAGS
FLAGS(sys.argv)
flags.DEFINE_bool("render", False, "Whether to render with pygame.")

env_names = ["DefeatZerglingsAndBanelings", "DefeatRoaches",
             "CollectMineralShards", "MoveToBeacon", "FindAndDefeatZerglings",
             "BuildMarines", "CollectMineralsAndGas"]

rl_algo = 'ppo'


def main(_):
    for map_name in env_names:
        actor = ActorNet()
        critic = CriticNet()

        if rl_algo == 'ddpg':
            memory = Memory(limit=arglist.memory_limit,
                            action_shape=arglist.action_shape,
                            observation_shape=arglist.observation_shape)
            learner = DDPGAgent(actor, critic, memory)

        elif rl_algo == 'ppo':
            memory = EpisodeMemory(limit=arglist.memory_limit,
                                   action_shape=arglist.action_shape,
                                   observation_shape=arglist.observation_shape)
            learner = PPOAgent(actor, critic, memory)

        preprocess = Preprocess()
        game = MiniGame(map_name, learner, preprocess, nb_episodes=10000)
        game.run()
    return 0


if __name__ == '__main__':
    app.run(main)
