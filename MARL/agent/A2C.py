import random

import numpy as np
import torch as th
from torch import nn
from torch.optim import Adam, RMSprop

from MARL.common.Memory import ReplayMemory
from MARL.common.Model import CriticNetwork, ActorNet
from MARL.common.utils import entropy, index_to_one_hot, to_tensor_var


class A2C:
    """
    A2C agent
    using pytorch model approximation based method
    - take exploration action, expect epsilon value or use decay_epsilon()
    - save experiences to replay memory
    - train actor critic model
    - Actor takes state as input
    - Critic takes both state and action as input
    - save model
    """

    def __init__(self, state_dim, action_dim,
                 memory_capacity=10000,
                 reward_gamma=0.99, reward_scale=1.,
                 actor_hidden_size=32, critic_hidden_size=32,
                 actor_output_act=nn.functional.softmax, critic_loss="mse",
                 actor_lr=0.001, critic_lr=0.001,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=0.003,
                 marl_training_strategy="concurrent", use_cuda=True):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale

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
        self.target_tau = 0.01

        # params for epsilon greedy
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.use_cuda = use_cuda and th.cuda.is_available()
        self.marl_training_strategy = marl_training_strategy

        self.actor = ActorNet(self.state_dim, actor_hidden_size, self.action_dim)
        self.critic = CriticNetwork(self.state_dim, self.action_dim, self.critic_hidden_size, 1)

        if self.optimizer_type == "adam":
            self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
        elif self.optimizer_type == "rmsprop":
            self.actor_optimizer = RMSprop(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = RMSprop(self.critic.parameters(), lr=self.critic_lr)
        if self.use_cuda:
            self.actor.cuda()

    # train on a roll_out batch
    def learn(self):
        """
        Note:
            - use learn() after pushing some experiences to the replay memory
            - if the environment is multi-agent with centralized training consider use shared_learning()
        """

        assert self.marl_training_strategy == "concurrent"

        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.state_dim)
        one_hot_actions = index_to_one_hot(batch.actions, self.action_dim)
        actions_var = to_tensor_var(one_hot_actions, self.use_cuda).view(-1, self.action_dim)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, 1)

        # update actor network
        self.actor_optimizer.zero_grad()
        action_log_probs = self.actor(states_var)
        entropy_loss = th.mean(entropy(th.exp(action_log_probs)))
        action_log_probs = th.sum(action_log_probs * actions_var, 1)
        values = self.critic(states_var, actions_var)
        advantages = rewards_var - values.detach()
        pg_loss = -th.mean(action_log_probs * advantages)
        actor_loss = pg_loss - entropy_loss * self.entropy_reg
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # update critic network
        self.critic_optimizer.zero_grad()
        target_values = rewards_var

        if self.critic_loss == "huber":
            critic_loss = nn.functional.smooth_l1_loss(values, target_values)
        else:
            critic_loss = nn.MSELoss()(values, target_values)

        critic_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

    def shared_learning(self,
                        n_agents,
                        agent_index,
                        shared_critic,
                        shared_critic_optim,
                        shared_batch_sample):
        """
         centralized learning for N agents
         update and synchronize the shared critic network parameters during the learning process.
         @mahmoudtaouti
        """

        assert self.marl_training_strategy == "centralized"
        states_var = to_tensor_var(shared_batch_sample.states, self.use_cuda).view(-1, n_agents, self.state_dim)
        one_hot_actions = [index_to_one_hot(a, self.action_dim) for a in shared_batch_sample.actions]
        actions_var = to_tensor_var(one_hot_actions, self.use_cuda).view(-1, n_agents, self.action_dim)
        rewards_var = to_tensor_var(shared_batch_sample.rewards, self.use_cuda).view(-1, n_agents, 1)

        whole_states_var = states_var.view(-1, n_agents * self.state_dim)
        whole_actions_var = actions_var.view(-1, n_agents * self.action_dim)

        # update actor network
        self.actor_optimizer.zero_grad()
        action_log_probs = self.actor(states_var[:, agent_index, :])
        entropy_loss = th.mean(entropy(th.exp(action_log_probs)))
        action_log_probs = th.sum(action_log_probs * actions_var[:, agent_index, :], 1)

        values = shared_critic(whole_states_var, whole_actions_var)

        # calculate advantages
        advantages = rewards_var[:, agent_index, :] - values.detach()
        pg_loss = -th.mean(action_log_probs * advantages)
        actor_loss = pg_loss - entropy_loss * self.entropy_reg
        actor_loss.backward()

        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # update critic network
        shared_critic_optim.zero_grad()
        target_values = rewards_var[:, agent_index, :]

        critic_loss = nn.MSELoss()(values, target_values)

        critic_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(shared_critic.parameters(), self.max_grad_norm)
        shared_critic_optim.step()

    # predict softmax action based on state
    def _softmax_action(self, state):
        state_var = to_tensor_var([state], self.use_cuda)
        softmax_action_var = th.exp(self.actor(state_var))
        # dist = th.distributions.Categorical(probs=state_var)
        # dist = dist.sample()
        # action = dist.detach().data.numpy()[0]
        if self.use_cuda:
            softmax_action = softmax_action_var.data.cpu().numpy()[0]
        else:
            softmax_action = softmax_action_var.data.numpy()[0]

        return softmax_action

    # choose an action based on state with random noise added for exploration in training
    def exploration_action(self, state, epsilon=None):

        if epsilon:
            self.epsilon = epsilon
        else:
            self.decay_epsilon()

        softmax_action = self._softmax_action(state)
        # the epsilon greedy is calculated with MARL
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_dim)
        else:
            action = np.argmax(softmax_action)
        return action

    # choose an action based on state for execution
    def action(self, state):
        softmax_action = self._softmax_action(state)
        action = np.argmax(softmax_action)
        return action

    def remember(self, state, action, reward, new_state, done):
        self.memory.push(state, action, reward, new_state, done)

    def decay_epsilon(self):
        # decrease the exploration rate epsilon over time
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_end, self.epsilon)

    # evaluate value for a state-action pair
    def value(self, state, action):
        state_var = to_tensor_var([state], self.use_cuda)
        action = index_to_one_hot(action, self.action_dim)
        action_var = to_tensor_var([action], self.use_cuda)
        value_var = self.critic(state_var, action_var)
        if self.use_cuda:
            value = value_var.data.cpu().numpy()[0]
        else:
            value = value_var.data.numpy()[0]
        return value

    def save(self, global_step, out_dir='/model'):
        actor_file_path = out_dir + f"/actor_.pt"
        critic_file_path = out_dir + f"/critic_.pt"

        th.save({'global_step': global_step,
                 'model_state_dict': self.actor.state_dict(),
                 'optimizer_state_dict': self.actor_optimizer.state_dict()},
                actor_file_path)
        th.save({'global_step': global_step,
                 'model_state_dict': self.critic.state_dict(),
                 'optimizer_state_dict': self.critic_optimizer.state_dict()},
                critic_file_path)
