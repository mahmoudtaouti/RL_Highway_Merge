import random

import torch.nn as nn
from torch.optim import Adam, RMSprop
import torch as th

import numpy as np
from copy import deepcopy

from MARL.common.Memory import ReplayMemory
from MARL.common.Model import ActorNetwork, CriticNetwork
from MARL.common.utils import to_tensor_var, exponential_epsilon_decay

# seed
seed = 10
np.random.seed(seed)
th.manual_seed(seed)
random.seed(seed)


class DDPG:
    """
    An agent learned with Deep Deterministic Policy Gradient using Actor-Critic framework
    - Actor takes state as input
    - Critic takes both state and action as input
    - Critic uses gradient temporal-difference learning
    """

    def __init__(self, state_dim, action_dim,
                 memory_capacity=10000, max_steps=None,
                 target_tau=0.01, target_update_steps=5,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_hidden_size=32, critic_hidden_size=32,
                 actor_output_act=nn.functional.tanh, critic_loss="mse",
                 actor_lr=0.001, critic_lr=0.001,
                 optimizer_type="adam", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=200,
                 use_cuda=False):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_episodes = 0
        self.n_steps = 0
        self.max_steps = max_steps
        self.roll_out_n_steps = 1

        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale
        self.done_penalty = done_penalty

        self.memory = ReplayMemory(memory_capacity)
        self.actor_hidden_size = actor_hidden_size
        self.critic_hidden_size = critic_hidden_size
        self.actor_output_act = actor_output_act
        self.critic_loss = critic_loss
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optimizer_type = optimizer_type
        self.entropy_reg = entropy_reg
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.episodes_before_train = episodes_before_train
        self.target_tau = 0.01

        # params for epsilon greedy
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.use_cuda = use_cuda and th.cuda.is_available()

        self.target_tau = target_tau
        self.target_update_steps = target_update_steps

        self.actor = ActorNetwork(self.state_dim, self.actor_hidden_size, self.action_dim, self.actor_output_act)
        self.critic = CriticNetwork(self.state_dim, self.action_dim, self.critic_hidden_size, 1)
        # to ensure target network and learning network has the same weights
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

        if self.optimizer_type == "adam":
            self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
        elif self.optimizer_type == "rmsprop":
            self.actor_optimizer = RMSprop(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = RMSprop(self.critic.parameters(), lr=self.critic_lr)

        if self.use_cuda:
            self.actor.cuda()
            self.critic.cuda()
            self.actor_target.cuda()
            self.critic_target.cuda()

    # train on a sample batch
    def train(self, batch_sample):
        """
         Do not train until exploration is enough
        """

        batch = self.memory.sample(self.batch_size)
        state_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.state_dim)
        action_var = to_tensor_var(batch.actions, self.use_cuda).view(-1, self.action_dim)
        reward_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, 1)
        next_state_var = to_tensor_var(batch.next_states, self.use_cuda).view(-1, self.state_dim)
        done_var = to_tensor_var(batch.dones, self.use_cuda).view(-1, 1)

        # estimate the target q with actor_target network and critic_target network
        next_action_var = self.actor_target(next_state_var)
        next_q = self.critic_target(next_state_var, next_action_var).detach()
        target_q = self.reward_scale * reward_var + self.reward_gamma * next_q * (1. - done_var)

        # update critic network
        self.critic_optimizer.zero_grad()
        # current Q values
        current_q = self.critic(state_var, action_var)
        # rewards is target Q values
        if self.critic_loss == "huber":
            critic_loss = nn.functional.smooth_l1_loss(current_q, target_q)
        else:
            critic_loss = nn.MSELoss()(current_q, target_q)
        critic_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # update actor network
        self.actor_optimizer.zero_grad()
        # the accurate action prediction
        action = self.actor(state_var)
        # actor_loss is used to maximize the Q value for the predicted action
        actor_loss = - self.critic(state_var, action)
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # update actor target network and critic target network
        if self.n_steps % self.target_update_steps == 0 and self.n_steps > 0:
            self._soft_update_target(self.critic_target, self.critic)
            self._soft_update_target(self.actor_target, self.actor)

    # choose an action based on state with random noise added for exploration in training
    def exploration_action(self, state):
        action = self.action(state)
        epsilon = exponential_epsilon_decay(
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            decay_rate=self.epsilon_decay,
            episode=self.n_episodes)
        # add noise
        noise = np.random.randn(self.action_dim) * epsilon
        action += noise
        return action

    # choose an action based on state for execution
    def action(self, state):
        action_var = self.actor(to_tensor_var([state], self.use_cuda))
        if self.use_cuda:
            action = action_var.data.cpu().numpy()[0]
        else:
            action = action_var.data.numpy()[0]
        return action

    # discount roll out rewards
    def _discount_reward(self, rewards, final_value):
        discounted_r = np.zeros_like(rewards)
        running_add = final_value
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.reward_gamma + rewards[t]
            discounted_r[t] = running_add
        return discounted_r

    # soft update the actor target network or critic target network
    def _soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(
                (1. - self.target_tau) * t.data + self.target_tau * s.data)

    def save(self, out_dir='/model'):
        pass
