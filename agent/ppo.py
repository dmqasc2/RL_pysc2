import torch
import shutil
import copy
from torch.nn.functional import gumbel_softmax
from utils import arglist
from agent.agent import Agent


class PPOAgent(Agent):
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
                    x = torch.tensor(replays[key][subkey], dtype=torch.float32)
                    batch[key][subkey] = x.to(self.device)
            else:
                # process
                x = torch.tensor(replays[key], dtype=torch.float32)
                x = torch.squeeze(x)
                batch[key] = x.to(self.device)

        return batch['obs0'], batch['actions'], batch['rewards'], batch['obs1'], batch['terminals1']

    @staticmethod
    def gumbel_softmax_hard(x):
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

    def get_gae(self, rewards, masks, values):
        # mask: terminal
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        returns = torch.zeros_like(rewards)
        advants = torch.zeros_like(rewards)

        running_returns = 0
        previous_value = 0
        running_advants = 0

        for t in reversed(range(0, len(rewards))):
            running_returns = rewards[t] + arglist.gamma * running_returns * masks[t]
            running_tderror = rewards[t] + arglist.gamma * previous_value * masks[t] - \
                              values.data[t]
            running_advants = running_tderror + arglist.gamma * arglist.lamda * \
                              running_advants * masks[t]

            returns[t] = running_returns
            previous_value = values.data[t]
            advants[t] = running_advants

        advants = (advants - advants.mean()) / advants.std()
        return returns, advants

    def surrogate_loss(self, actor, advants, states, old_policy, actions, index):
        mu, std, logstd = actor(torch.Tensor(states))
        new_policy = log_density(actions, mu, std, logstd)
        old_policy = old_policy[index]

        ratio = torch.exp(new_policy - old_policy)
        surrogate = ratio * advants
        return surrogate, ratio

    def optimize(self):
        """
        Samples a random batch from replay memory and performs optimization
        :return:
        """
        if self.memory.nb_entries < arglist.BatchSize * 10:
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
        self.soft_update(self.target_actor, self.actor, arglist.TAU)
        self.soft_update(self.target_critic, self.critic, arglist.TAU)
        return loss_actor, loss_critic


        ###PPO
        values = self.critic(torch.Tensor(states))

        # ----------------------------
        # step 1: get returns and GAEs and log probability of old policy
        returns, advants = self.get_gae(rewards, masks, values)
        mu, std, logstd = self.actor(torch.Tensor(states))
        old_policy = log_density(torch.Tensor(actions), mu, std, logstd)

        criterion = torch.nn.MSELoss()
        n = len(states)
        arr = np.arange(n)

        # ----------------------------
        # step 2: get value loss and actor loss and update actor & critic
        for epoch in range(10):
            np.random.shuffle(arr)

            for i in range(n // hp.batch_size):
                batch_index = arr[hp.batch_size * i: hp.batch_size * (i + 1)]
                batch_index = torch.LongTensor(batch_index)
                inputs = torch.Tensor(states)[batch_index]
                returns_samples = returns.unsqueeze(1)[batch_index]
                advants_samples = advants.unsqueeze(1)[batch_index]
                actions_samples = torch.Tensor(actions)[batch_index]

                loss, ratio = self.surrogate_loss(actor, advants_samples, inputs,
                                             old_policy.detach(), actions_samples,
                                             batch_index)

                values = self.critic(inputs)
                critic_loss = criterion(values, returns_samples)
                critic_optim.zero_grad()
                critic_loss.backward()
                critic_optim.step()

                clipped_ratio = torch.clamp(ratio,
                                            1.0 - hp.clip_param,
                                            1.0 + hp.clip_param)
                clipped_loss = clipped_ratio * advants_samples
                actor_loss = -torch.min(loss, clipped_loss).mean()

                actor_optim.zero_grad()
                actor_loss.backward()
                actor_optim.step()
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
