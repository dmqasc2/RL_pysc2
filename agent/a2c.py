import torch
import numpy as np
import shutil
import copy
from utils import arglist


class LearnerSeperatedAC:
    def __init__(self, actor, critic, memory):
        """
        DDPG learning for seperated actor & critic networks.
        """
        self.device = arglist.DEVICE

        self.iter = 0
        self.actor = actor.to(self.device)
        self.target_actor = copy.deepcopy(actor).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), arglist.actor_learning_rate)

        self.critic = critic.to(self.device)
        self.target_critic = copy.deepcopy(critic).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), arglist.critic_learning_rate)

        self.memory = memory

        self.target_actor.eval()
        self.target_critic.eval()

    def preprocess_available_actions(self, available_actions, max_action=len(Actions.FUNCTIONS)):
        a_actions = np.zeros((max_action), dtype='float32')
        a_actions[available_actions] = 1.
        return a_actions

    def select_action(self, minimap_vb, screen_vb, valid_action_vb):
        valid_action_vb = self.preprocess_available_actions(valid_action_vb, max_action=len(Actions.FUNCTIONS))

        minimap_vb = minimap_vb.astype('float32')
        screen_vb = screen_vb.astype('float32')

        minimap_vb = torch.from_numpy(np.expand_dims(minimap_vb, 0)).to(arglist.DEVICE)
        screen_vb = torch.from_numpy(np.expand_dims(screen_vb, 0)).to(arglist.DEVICE)
        valid_action_vb = torch.from_numpy(np.expand_dims(valid_action_vb, 0)).to(arglist.DEVICE)

        probs_spatial, probs_nonspatial = self.policy.forward(minimap_vb, screen_vb, valid_action_vb)
        # spatial action
        action_spatial = torch.argmax(probs_spatial, dim=-1)
        self.saved_log_probs_spatial.append(probs_spatial[0][action_spatial.item()].log())
        # non-spatial action
        action_nonspatial = torch.argmax(probs_nonspatial, dim=-1)
        self.saved_log_probs_nonspatial.append(probs_nonspatial[0][action_nonspatial.item()].log())

    def optimize(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + arglist.EPS)
        for log_prob_spatial, log_prob_nonspatial, R in zip(self.saved_log_probs_spatial,
                                                            self.saved_log_probs_nonspatial, returns):
            policy_loss.append((-log_prob_spatial * R) + (-log_prob_nonspatial * R))
        self.optimizer.zero_grad()
        policy_loss = torch.tensor(policy_loss, requires_grad=True).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs_nonspatial[:]
        del self.saved_log_probs_spatial[:]

    def save_models(self, episode_count):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:
        """
        torch.save(self.policy.state_dict(), './Models/' + str(episode_count) + '_reinforce.pt')
        print('Models saved successfully')

    def load_models(self, episode):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        self.policy.load_state_dict(torch.load('./Models/' + str(episode) + '_reinforce.pt'))
        print('Models loaded succesfully')

    def save_training_checkpoint(self, state, is_best, episode_count):
        """
        Saves the models, with all training parameters intact
        :param state:
        :param is_best:
        :param filename:
        :return:
        """
        filename = str(episode_count) + 'checkpoint.path.rar'
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')