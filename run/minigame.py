from pysc2.env import sc2_env
from pysc2.lib import actions
from utils import arglist


agent_format = sc2_env.AgentInterfaceFormat(
    feature_dimensions=sc2_env.Dimensions(
        screen=(arglist.FEAT2DSIZE, arglist.FEAT2DSIZE),
        minimap=(arglist.FEAT2DSIZE, arglist.FEAT2DSIZE), )
)


class MiniGame(sc2_env.SC2Env):
    def __init__(self, map_name, learner, nb_episodes=1000):
        super(MiniGame, self).__init__()
        self.map_name = map_name
        self.nb_max_steps = 2000
        self.nb_episodes = nb_episodes
        self.env = sc2_env.SC2Env(map_name=self.map_name,
                                  step_mul=16,
                                  visualize=False,
                                  agent_interface_format=[agent_format])
        self.learner = learner
        
    
    def post_preprocessing_action(self, actions):
        """
        Transform selected non_spatial and spatial actions into pysc2 FunctionCall
        Args:
            non_spatial_action: ndarray, shape (1, 1)
            spatial_action: ndarray, shape (1, 1)
        Returns:
            FunctionCall as action for pysc2_env
        """
        act_id = actions[1]
        target = actions[0]
        target_point = [
            int(target % arglist.FEAT2DSIZE),
            int(target // arglist.FEAT2DSIZE)
        ]  # (x, y)

        act_args = []
        for arg in actions.FUNCTIONS[act_id].args:
            if arg.name in ('screen', 'minimap', 'screen2'):
                act_args.append(target_point)
            else:
                act_args.append([0])
        return actions.FunctionCall(act_id, act_args)

    def run(self):
        reward_cumulative = []
        for i_episode in range(self.nb_episodes):
            state = self.env.reset()[0]
            for t in range(1, self.nb_max_steps):  # Don't infinite loop while learning
                actions = self.learner.select_action(state.observation['feature_screen'],
                                                     state.observation['feature_minimap'],
                                                     state.observation['available_actions'])
                actions = self.post_preprocessing_action(actions)
                # print(actions)
                state = self.env.step(actions=[actions])[0]
                self.learner.rewards.append(state.reward)
                if state.last():
                    cum_reward = state.observation["score_cumulative"]
                    reward_cumulative.append(cum_reward[0])
                    break
            self.learner.optimize()

        self.env.close()
        print(reward_cumulative)
        print(np.mean(reward_cumulative))