import torch
import shutil
import copy
from torch.nn.functional import gumbel_softmax
from utils import arglist
from agent.agent import Agent


class DDPGAgent(Agent):
    def __init__(self, actor, critic, memory):
        """
        DDPG learning for seperated actor & critic networks.
        """
        self.device = arglist.DEVICE
        self.nb_actions = arglist.NUM_ACTIONS

        self.iter = 0
        self.actor = actor.to(self.device)
        self.target_actor = copy.deepcopy(actor).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), arglist.DDPG.LEARNINGRATE)

        self.critic = critic.to(self.device)
        self.target_critic = copy.deepcopy(critic).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), arglist.DDPG.LEARNINGRATE)

        self.memory = memory

        self.target_actor.eval()
        self.target_critic.eval()

    def process_batch(self):
        """
        Transforms numpy replays to torch tensor
        :return: dict of torch.tensor
        """
        replays = self.memory.sample(arglist.DDPG.BatchSize)

        # initialize batch experience
        batch = {'state0': {'minimap': [], 'screen': [], 'nonspatial': []},
                 'action': {'categorical': [], 'screen1': [], 'screen2': []},
                 'reward': [],
                 'state1': {'minimap': [], 'screen': [], 'nonspatial': []},
                 'terminal1': [],
                 }
        # append experience to list
        for e in replays:
            # state0
            for k, v in e.state0[0].items():
                batch['state0'][k].append(v)
            # action
            for k, v in e.action.items():
                batch['action'][k].append(v)
            # reward
            batch['reward'].append(e.reward)
            # state1
            for k, v in e.state1[0].items():
                batch['state1'][k].append(v)
            # terminal1
            batch['terminal1'].append(0. if e.terminal1 else 1.)

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

        return batch['state0'], batch['action'], batch['reward'], batch['state1'], batch['terminal1']

    def gumbel_softmax_hard(self, x):
        shape = x.shape
        if len(shape) == 4:
            # merge batch and seq dimensions
            x_reshape = x.contiguous().view(shape[0], -1)
            y = torch.nn.functional.gumbel_softmax(x_reshape, hard=True, dim=-1)
            # We have to reshape Y
            y = y.contiguous().view(shape)
        else:
            y = torch.nn.functional.gumbel_softmax(x, hard=True, dim=-1)

        return y

    def optimize(self):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """
        if self.memory.nb_entries < arglist.DDPG.BatchSize * 10:
            return 0, 0

        s0, a0, r, s1, d = self.process_batch()
        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        logits1 = self.target_actor.forward(s1)
        a1 = {'categorical': 0, 'screen1': 0, 'screen2': 0}
        for key, value in logits1.items():
            a1[key] = self.gumbel_softmax_hard(value)
        q_next = self.target_critic.forward(s1, a1)
        q_next = q_next.detach()
        q_next = torch.squeeze(q_next)
        # Loss: TD error
        # y_exp = r + gamma*Q'( s1, pi'(s1))
        y_expected = r + arglist.DDPG.GAMMA * q_next * (1. - d)
        # y_pred = Q( s0, a0)
        y_predicted = self.critic.forward(s0, a0)
        y_predicted = torch.squeeze(y_predicted)

        # Sum. Loss
        loss_critic = torch.nn.SmoothL1Loss()(y_predicted, y_expected)

        # Update critic
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # ---------------------- optimize actor ----------------------
        pred_logits0 = self.actor.forward(s0)
        pred_a0 = {'categorical': 0, 'screen1': 0, 'screen2': 0}
        for key, value in pred_logits0.items():
            pred_a0[key] = self.gumbel_softmax_hard(value)

        # Loss: regularization
        l2_reg = torch.cuda.FloatTensor(1)
        for W in self.actor.parameters():
            l2_reg = l2_reg + W.norm(2)

        # Loss: max. Q
        Q = self.critic.forward(s0, pred_a0)
        actor_maxQ = -1 * Q.mean()

        # Sum. Loss
        loss_actor = actor_maxQ
        loss_actor += torch.squeeze(l2_reg) * 1e-3

        # Update actor
        # runs random noise to exploration
        self.actor.train()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        # Update target env
        self.soft_update(self.target_actor, self.actor, arglist.DDPG.TAU)
        self.soft_update(self.target_critic, self.critic, arglist.DDPG.TAU)
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

        def save_models(self, fname, save_dir='results'):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:
        """
        os.makedirs(save_dir, exist_ok=True)

        actor_file = os.path.join(save_dir, str(fname) + '_actor.pt')
        critic_file = os.path.join(save_dir, str(fname) + '_critic.pt')

        torch.save(self.target_actor.state_dict(), actor_file)
        torch.save(self.target_critic.state_dict(), critic_file)
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
