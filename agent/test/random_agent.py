#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.agents import random_agent
from pysc2.env import sc2_env
from pysc2.env.environment import StepType

from absl import app
from absl import flags
import sys

import numpy as np

nb_episodes = 10
nb_max_steps = 2000

FLAGS = flags.FLAGS
FLAGS(sys.argv)
flags.DEFINE_bool("render", False, "Whether to render with pygame.")

agent_format = sc2_env.AgentInterfaceFormat(
    feature_dimensions=sc2_env.Dimensions(
        screen=(32, 32),
        minimap=(32, 32),)
)

env_names = ["DefeatZerglingsAndBanelings", "DefeatRoaches",
             "CollectMineralShards", "MoveToBeacon", "FindAndDefeatZerglings",
             "BuildMarines", "CollectMineralsAndGas"]


def run(env_name):
    env = sc2_env.SC2Env(
        map_name=env_name,  # "BuildMarines",
        step_mul=16,
        visualize=False,
        agent_interface_format=[agent_format])

    agent = random_agent.RandomAgent()
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    agent.setup(observation_spec[0], action_spec[0])

    reward = []
    reward_cumulative = []
    for ep in range(nb_episodes):
        reward.append(0)
        obs = env.reset()[0]
        '''        
        obs[0]  # step_type
        obs[1]  # reward
        obs[2]  # discount
        obs[3]  # observation        
        '''
        agent.reset()
        while True:
            actions = agent.step(obs)
            obs = env.step(actions=[actions])[0]
            reward[-1] += obs.reward
            if obs.last():
                cum_reward = obs.observation["score_cumulative"]
                reward_cumulative.append(cum_reward[0])
                break
    env.close()
    print(reward)
    print(np.mean(reward))
    print(reward == reward_cumulative)


def main(_):
    for e in env_names:
        run(e)
    return 0


if __name__ == '__main__':
    app.run(main)






