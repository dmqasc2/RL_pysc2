from absl import app
from absl import flags
import sys
import torch
from utils import arglist
from runs.minigame import MiniGame
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
        if rl_algo == 'ddpg':
            from agent.ddpg import DDPGAgent
            from networks.acnetwork_q_seperated import ActorNet, CriticNet
            from utils.memory import Memory

            actor = ActorNet()
            critic = CriticNet()
            memory = Memory(limit=arglist.memory_limit,
                            action_shape=arglist.action_shape,
                            observation_shape=arglist.observation_shape)
            learner = DDPGAgent(actor, critic, memory)

        elif rl_algo == 'ppo':
            from agent.ppo import PPOAgent
            from networks.acnetwork_v_seperated import ActorNet, CriticNet
            from utils.memory import EpisodeMemory

            actor = ActorNet()
            critic = CriticNet()
            memory = EpisodeMemory(limit=arglist.memory_limit,
                                   action_shape=arglist.action_shape,
                                   observation_shape=arglist.observation_shape)
            learner = PPOAgent(actor, critic, memory)

        else:
            raise NotImplementedError()

        preprocess = Preprocess()
        game = MiniGame(map_name, learner, preprocess, nb_episodes=10000)
        game.run()
    return 0


if __name__ == '__main__':
    app.run(main)
