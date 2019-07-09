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
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), arglist.LEARNINGRATE)

        self.critic = critic.to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), arglist.LEARNINGRATE)

        self.memory = memory

    def process_batch(self):
        """
        Transforms numpy replays to torch tensor
        :return: dict of torch.tensor
        """
        replays = self.memory.sample(arglist.BatchSize)
        batch = {}
        for key in replays:
            batch[key] = {}
            if type(replays[key]) is dict:
                for subkey in replays[key]:
                    # process
                    x = torch.tensor(replays[key][subkey], dtype=torch.float32)
                    batch[key][subkey] = x.to(self.device)
            else:
                # process
                x = torch.tensor(replays[key], dtype=torch.float32)
                x = torch.squeeze(x)
                batch[key] = x.to(self.device)

        return batch['obs0'], batch['actions'], batch['rewards'], batch['obs1'], batch['terminals1']

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
        '''
        rewards: immediate rewards
        masks: terminals
        values: value function from critic network
        '''
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        running_returns = 0
        previous_value = 0
        running_advantages = 0

        for t in reversed(range(0, len(rewards))):
            running_returns = rewards[t] + arglist.PPOHyperParams.gamma * \
                              running_returns * masks[t]
            running_tderror = rewards[t] + arglist.PPOHyperParams.gamma * \
                              previous_value * masks[t] - values.data[t]
            running_advantages = running_tderror + arglist.PPOHyperParams.gamma * \
                                 arglist.PPOHyperParams.lamda * running_advantages * masks[t]

            returns[t] = running_returns
            previous_value = values.data[t]
            advantages[t] = running_advantages

        advantages = (advantages - advantages.mean()) / advantages.std()
        return returns, advantages

    def surrogate_loss(self, advantages, obs, old_policy, actions, index):
        '''
        <arguments>
            index: batch index
        <original: contituous action space using tensorflow>
            dist = tf.distributions.Categorical(probs=tf.nn.softmax(policy))
            new_policy = log_density(actions, mu, std, logstd)

        <fix: PPO for discrete action space>
            reference: https://github.com/takuseno/ppo/blob/master/build_graph.py
        '''
        logits = self.actor(obs)
        surrogate = {'categorical': 0, 'screen1': 0, 'screen2': 0}
        ratio = {'categorical': 0, 'screen1': 0, 'screen2': 0}
        for key, value in actions.items():
            # get probability from logits using Gumbel softmax (for exploration)
            probs = self.gumbel_softmax(logits[key], hard=False)
            # make distribution
            m = Categorical(probs)
            
            # calc. ratio
            new_policy = m.log_prob(actions[key])
            old_policy = old_policy[key][index]
            r = torch.exp(new_policy - old_policy)
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

        self.actor.train()
        self.critic.train()

        s0, a0, r, _, d = self.process_batch()
        values = self.critic(s0)

        # ----------------------------
        # step 1: get returns and GAEs and log probability of old policy
        returns, advantages = self.get_gae(r, d, values)

        logits = self.actor(s0)
        old_policy = {'categorical': 0, 'screen1': 0, 'screen2': 0}
        for k, v in logits.items():
            # get probability from logits using Gumbel softmax (for exploration)
            probs = self.gumbel_softmax(v, hard=False)
            # make distribution
            m = Categorical(probs)
            old_policy[k] = m.log_prob(a0[k]).detach()

        criterion = torch.nn.MSELoss()
        n = len(r)
        arr = np.arange(n)

        # ----------------------------
        # step 2: get value loss and actor loss and update actor & critic
        for epoch in range(10):
            np.random.shuffle(arr)

            for i in range(n // arglist.BatchSize):
                batch_index = arr[arglist.BatchSize * i: arglist.BatchSize * (i + 1)]
                batch_index = torch.LongTensor(batch_index)
                inputs = s0[batch_index]
                returns_samples = returns.unsqueeze(1)[batch_index]
                advantages_samples = advantages.unsqueeze(1)[batch_index]
                actions_samples = a0[batch_index]

                loss, ratio = self.surrogate_loss(advantages_samples, inputs,
                                                  old_policy, actions_samples,
                                                  batch_index)

                values = self.critic(inputs)
                loss_critic = criterion(values, returns_samples)
                self.critic_optimizer.zero_grad()
                loss_critic.backward()
                self.critic_optimizer.step()

                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)

                clipped_ratio = torch.clamp(ratio,
                                            1.0 - arglist.PPOHyperParams.clip_param,
                                            1.0 + arglist.PPOHyperParams.clip_param)
                clipped_loss = clipped_ratio * advantages_samples
                loss_actor = -torch.min(loss, clipped_loss).mean()

                self.actor_optimizer.zero_grad()
                loss_actor.backward()
                self.actor_optimizer.step()

        return loss_actor, loss_critic

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
        torch.save(self.target_actor.state_dict(), str(fname) + '_actor.pt')
        torch.save(self.target_critic.state_dict(), str(fname) + '_critic.pt')
        print('Models saved successfully')

    def load_models(self, fname):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        self.actor.load_state_dict(torch.load(str(fname) + '_actor.pt'))
        self.critic.load_state_dict(torch.load(str(fname) + '_critic.pt'))
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)
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
