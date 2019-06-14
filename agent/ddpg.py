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
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), arglist.LEARNINGRATE)

        self.critic = critic.to(self.device)
        self.target_critic = copy.deepcopy(critic).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), arglist.LEARNINGRATE)

        self.memory = memory

        self.target_actor.eval()
        self.target_critic.eval()

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
                    batch[key][subkey] = torch.tensor(replays[key][subkey], dtype=torch.float32).to(self.device)
            else:
                # process
                batch[key] = torch.tensor(replays[key], dtype=torch.float32).to(self.device)

        return batch['obs0'], batch['actions'], batch['rewards'], batch['obs1'], batch['terminals1']

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
        s0, a0, r, s1, d = self.process_batch()
        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        logits1 = self.target_actor.forward(s1)
        a1 = [self.gumbel_softmax_hard(x) for x in logits1]

        q_next = self.target_critic.forward(s1, a1)
        q_next = q_next.detach()
        q_next = torch.squeeze(q_next)
        # Loss: TD error
        # y_exp = r + gamma*Q'( s1, pi'(s1))
        y_expected = r + arglist.GAMMA * q_next * (1. - d)
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
        pred_a0 = [self.gumbel_softmax_hard(x) for x in pred_logits0]

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
        self.soft_update(self.target_actor, self.actor, arglist.TAU)
        self.soft_update(self.target_critic, self.critic, arglist.TAU)

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
