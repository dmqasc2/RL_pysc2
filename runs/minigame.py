from pysc2.env import sc2_env
from utils import arglist
from copy import deepcopy


agent_format = sc2_env.AgentInterfaceFormat(
    feature_dimensions=sc2_env.Dimensions(
        screen=(arglist.FEAT2DSIZE, arglist.FEAT2DSIZE),
        minimap=(arglist.FEAT2DSIZE, arglist.FEAT2DSIZE), )
)


class MiniGame:
    def __init__(self, map_name, learner, preprocess, nb_episodes=1000):
        self.map_name = map_name
        self.nb_max_steps = 2000
        self.nb_episodes = nb_episodes
        self.env = sc2_env.SC2Env(map_name=self.map_name,
                                  step_mul=16,
                                  visualize=False,
                                  agent_interface_format=[agent_format])
        self.learner = learner
        self.preprocess = preprocess

    def run_ddpg(self, is_training=True):
        reward_cumulative = []
        for i_episode in range(self.nb_episodes):
            state = self.env.reset()[0]
            for t in range(self.nb_max_steps):  # Don't infinite loop while learning
                obs = self.preprocess.get_observation(state)
                actions = self.learner.select_action(obs, valid_actions=obs['nonspatial'])
                state_new = self.env.step(actions=[actions])[0]

                # append memory
                actions = self.preprocess.postprocess_action(actions)
                self.learner.memory.append(obs, actions, state.reward, state.last(), training=is_training)
                # self.learner.optimize()

                if state.last():
                    cum_reward = state.observation["score_cumulative"]
                    reward_cumulative.append(cum_reward[0])
                    break
                state = deepcopy(state_new)
        self.env.close()
        print(reward_cumulative)

    def run_ppo(self, is_training=True):
        reward_cumulative = []
        for i_episode in range(self.nb_episodes):
            state = self.env.reset()[0]
            for t in range(self.nb_max_steps):  # Don't infinite loop while learning
                obs = self.preprocess.get_observation(state)
                actions = self.learner.select_action(obs, valid_actions=obs['nonspatial'])
                state_new = self.env.step(actions=[actions])[0]

                # append memory
                actions = self.preprocess.postprocess_action(actions)
                self.learner.memory.append(obs, actions, state.reward, state.last(), training=is_training)

                if state.last():
                    cum_reward = state.observation["score_cumulative"]
                    reward_cumulative.append(cum_reward[0])
                    break
                state = deepcopy(state_new)
            self.learner.optimize(update=True)
        self.env.close()
        print(reward_cumulative)

