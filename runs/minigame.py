import time
import os
from pysc2.env import sc2_env
from utils import arglist
from copy import deepcopy


agent_format = sc2_env.AgentInterfaceFormat(
    feature_dimensions=sc2_env.Dimensions(
        screen=(arglist.FEAT2DSIZE, arglist.FEAT2DSIZE),
        minimap=(arglist.FEAT2DSIZE, arglist.FEAT2DSIZE), )
)


class MiniGame:
    def __init__(self, map_name, learner, preprocess, nb_episodes=50000):
        self.map_name = map_name
        self.nb_max_steps = 200
        self.nb_episodes = nb_episodes
        self.env = sc2_env.SC2Env(map_name=self.map_name,
                                  step_mul=8,
                                  visualize=False,
                                  agent_interface_format=[agent_format])
        self.learner = learner
        self.preprocess = preprocess

    def write_history(self, fname, msg=None):
        fname = 'Models/' + fname

        if not os.path.exists(fname) or msg is None:
            f = open(fname, "w")
        else:
            f = open(fname, "a")
        f.write(str(msg) + '\n')
        f.close()

    def run_ddpg(self, is_training=True):
        cum_reward_best = 0
        self.write_history(self.map_name + '_history_ddpg.txt', msg=None)

        for i_episode in range(self.nb_episodes):
            state = self.env.reset()[0]
            for t in range(self.nb_max_steps):  # Don't infinite loop while learning
                obs = self.preprocess.get_observation(state)
                actions = self.learner.select_action(obs, valid_actions=obs['nonspatial'])
                state_new = self.env.step(actions=[actions])[0]

                # append memory
                actions = self.preprocess.postprocess_action(actions)
                self.learner.memory.append(obs, actions, state.reward, state.last(), training=is_training)

                self.learner.optimize(is_train=self.learner.iter % 8 == 0)

                if state.last():
                    cum_reward = state.observation["score_cumulative"][0]
                    # save networks
                    if cum_reward >= cum_reward_best:
                        self.learner.save_models(fname=self.map_name + '_ddpg')
                        cum_reward_best = cum_reward

                    # write cululative reward for every episodes
                    self.write_history(fname=self.map_name + '_history_ddpg.txt',
                                       msg='episde: {}, step: {}, score: {}'.format(i_episode, self.learner.iter, cum_reward))
                    break
                else:
                    state = deepcopy(state_new)

                self.learner.iter += 1
        self.env.close()

    def run_ppo(self, is_training=True):
        reward_cumulative = []
        f = open("PPO_result.txt", "w")
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
                    f = open("PPO_result.txt", "a")
                    cum_reward = state.observation["score_cumulative"]
                    reward_cumulative.append(cum_reward[0])
                    start = time.time()
                    self.learner.optimize(update=True)
                    self.learner.memory.clear()
                    end = time.time()
                    print(end-start)
                    f.write(f"score: [{cum_reward[0]}]\n")
                    break
                else:
                    state = deepcopy(state_new)

            time.sleep(0.5)

            if (i_episode + 1) % 10000 == 0:
                self.learner.save_models(fname=i_episode)

        self.env.close()
        f.close()
        print(reward_cumulative)

