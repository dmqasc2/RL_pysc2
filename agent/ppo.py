import torch
import shutil
from torch.nn.functional import gumbel_softmax
from torch.distributions import Categorical
from utils import arglist
from agent.agent import Agent
import numpy as np


class PPOAgent(Agent):
    def __init__(self, actor, critic, memory):
        """
        DDPG learning for seperated actor & critic networks.
        """
        self.device = arglist.DEVICE
        self.nb_actions = arglist.NUM_ACTIONS

        self.iter = 0
        self.actor = actor.to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), arglist.PPO.actor_lr)

        self.critic = critic.to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), arglist.PPO.critic_lr)

        self.memory = memory

    def process_batch(self):
        """
        Transforms numpy replays to torch tensor
        :return: dict of torch.tensor
        """
        replays = self.memory.sample()
        # initialize batch experience
        batch = {'state': {'minimap': [], 'screen': [], 'nonspatial': []},
                 'action': {'categorical': [], 'screen1': [], 'screen2': []},
                 'reward': [],
                 'terminal': [],
                 }
        # append experience to list
        for e in replays:
            # state0
            for k, v in e.state.items():
                batch['state'][k].append(v)
            # action
            for k, v in e.action.items():
                batch['action'][k].append(v)
            # reward
            batch['reward'].append(e.reward)
            # terminal1
            batch['terminal'].append(0. if e.terminal else 1.)

        # make torch tensor
        for key in batch.keys():
            if type(batch[key]) is dict:
                for subkey in batch[key]:
                    x = torch.tensor(batch[key][subkey], dtype=torch.float32)
                    batch[key][subkey] = x.to(self.device)
            else:
                x = torch.tensor(batch[key], dtype=torch.float32)
                x = torch.squeeze(x)
                batch[key] = x.to(self.device)

        return batch['state'], batch['action'], batch['reward'], batch['terminal']

    @staticmethod
    def flatten_actions(x):
        """
        process 2D actions to 1D actions (reshape actions)
        :input x: dict.
        :return: dict.
        """
        for k, v in x.items():
            if len(v.shape) == 4:
                x[k] = v.reshape(v.shape[0], -1)
        return x

    @staticmethod
    def gumbel_softmax(x, hard=True):
        shape = x.shape
        if len(shape) == 4:
            # merge batch and seq dimensions
            x_reshape = x.contiguous().view(shape[0], -1)
            y = torch.nn.functional.gumbel_softmax(x_reshape, hard=hard, dim=-1)
            # We have to reshape Y
            y = y.contiguous().view(shape)
        else:
            y = torch.nn.functional.gumbel_softmax(x, hard=hard, dim=-1)

        return y

    @staticmethod
    def get_gae(rewards, masks, values):
        """
        rewards: immediate rewards
        masks: terminals
        values: value function from critic network
        """
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        running_returns = 0
        previous_value = 0
        running_advantages = 0

        for t in reversed(range(0, len(rewards))):
            running_returns = rewards[t] + arglist.PPO.gamma * \
                              running_returns * masks[t]
            running_tderror = rewards[t] + arglist.PPO.gamma * \
                              previous_value * masks[t] - values.data[t]
            running_advantages = running_tderror + arglist.PPO.gamma * \
                                 arglist.PPO.lamda * running_advantages * masks[t]

            returns[t] = running_returns
            previous_value = values.data[t]
            advantages[t] = running_advantages

        advantages = (advantages - advantages.mean()) / advantages.std()
        return returns, advantages

    def surrogate_loss(self, advantages, obs, old_policy, actions, index):
        """
        <arguments>
            index: batch index
        <original: contituous action space using tensorflow>
            dist = tf.distributions.Categorical(probs=tf.nn.softmax(policy))
            new_policy = log_density(actions, mu, std, logstd)

        <fix: PPO for discrete action space>
            reference: https://github.com/takuseno/ppo/blob/master/build_graph.py
        """
        logits = self.actor(obs)
        logits = self.flatten_actions(logits)

        surrogate = {'categorical': 0, 'screen1': 0, 'screen2': 0}
        ratio = {'categorical': 0, 'screen1': 0, 'screen2': 0}
        for key, value in actions.items():
            # get probability from logits using Gumbel softmax (for exploration)
            probs = self.gumbel_softmax(logits[key], hard=False)
            # make distribution
            m = Categorical(probs)
            # calc. ratio
            new_policy = m.log_prob(actions[key].argmax(dim=-1))
            old_pol = old_policy[key][index]
            r = torch.exp(new_policy - old_pol)
            s = r * advantages

            ratio[key] = r
            surrogate[key] = s

        return surrogate, ratio

    def optimize(self, update=False):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """
        if not update:
            return 0, 0

        # PPO update
        self.actor.train()
        self.critic.train()

        s, a, r, d = self.process_batch()
        values = self.critic(s).reshape(-1)
        # ----------------------------
        # step 1: get returns and GAEs and log probability of old policy
        returns, advantages = self.get_gae(r, d, values)

        logits = self.actor(s)
        logits = self.flatten_actions(logits)

        old_policy = {'categorical': 0, 'screen1': 0, 'screen2': 0}
        for k, v in logits.items():
            # get probability from logits using Gumbel softmax (for exploration)
            probs = self.gumbel_softmax(v, hard=False)
            # make distribution
            m = Categorical(probs)
            old_policy[k] = m.log_prob(a[k].argmax(dim=-1)).detach()

        criterion = torch.nn.MSELoss()
        n = len(r)
        arr = np.arange(n)

        # ----------------------------
        # step 2: get value loss and actor loss and update actor & critic
        for epoch in range(10):
            np.random.shuffle(arr)

            for i in range(n // arglist.PPO.BatchSize):
                batch_index = arr[arglist.PPO.BatchSize * i: arglist.PPO.BatchSize * (i + 1)]
                # observation batch slicing
                inputs = {'minimap': 0, 'screen': 0, 'nonspatial': 0}
                for k in inputs.keys():
                    inputs[k] = s[k][batch_index]
                returns_samples = returns[batch_index]
                advantages_samples = advantages[batch_index]

                # action batch slicing
                actions_samples = {'categorical': 0, 'screen1': 0, 'screen2': 0}
                for k in actions_samples.keys():
                    actions_samples[k] = a[k][batch_index]

                loss, ratio = self.surrogate_loss(advantages_samples, inputs,
                                                  old_policy, actions_samples,
                                                  batch_index)

                values = self.critic(inputs).reshape(-1)
                loss_critic = criterion(values, returns_samples)
                self.critic_optimizer.zero_grad()
                loss_critic.backward()
                self.critic_optimizer.step()

                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)

                self.actor_optimizer.zero_grad()
                loss_actor = 0
                for k in loss.keys():
                    clipped_ratio = torch.clamp(ratio[k],
                                                1.0 - arglist.PPO.clip_param,
                                                1.0 + arglist.PPO.clip_param)
                    clipped_loss = clipped_ratio * advantages_samples
                    loss_actor += -torch.min(loss[k], clipped_loss).mean()
                loss_actor.backward()
                self.actor_optimizer.step()

        return loss_actor, loss_critic.item()

    def soft_update(self, target, source, tau):
        """
        Copies the parameters from source network (x) to target network (y) using the below update
        y = TAU*x + (1 - TAU)*y
        :param target: Target network (PyTorch)
        :param source: Source network (PyTorch)
        :return:
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def hard_update(self, target, source):
        """
        Copies the parameters from source network to target network
        :param target: Target network (PyTorch)
        :param source: Source network (PyTorch)
        :return:
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def save_models(self, fname):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:
        """
        torch.save(self.actor.state_dict(), str(fname) + '_actor.pt')
        torch.save(self.critic.state_dict(), str(fname) + '_critic.pt')
        print('Models saved successfully')

    def load_models(self, fname):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        self.actor.load_state_dict(torch.load(str(fname) + '_actor.pt'))
        self.critic.load_state_dict(torch.load(str(fname) + '_critic.pt'))
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
