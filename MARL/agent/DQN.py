import torch as th
from torch import nn
from torch.optim import Adam, RMSprop

import numpy as np
from MARL.common.Memory import ReplayMemory
from MARL.common.Model import ActorNetwork
from MARL.common.utils import identity, to_tensor_var


class DQN:
    """
    DQN agent
    using pytorch model approximation based method
    - take exploration action, expect epsilon value or use decay_epsilon()
    - save experiences to replay memory
    - train model and update values on batch sample
    - save model
    """
    def __init__(self, state_dim, action_dim,
                 memory_capacity=10000, batch_size=100,
                 reward_gamma=0.99, reward_scale=1.,
                 target_update_freq=30,
                 actor_hidden_size=128, actor_output_act=identity,
                 critic_loss="mse", actor_lr=0.001,
                 optimizer_type="rmsprop", max_grad_norm=0.5,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=0.001,
                 use_cuda=True):

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale

        self.memory = ReplayMemory(memory_capacity)
        self.actor_hidden_size = actor_hidden_size
        self.actor_output_act = actor_output_act
        self.critic_loss = critic_loss
        self.actor_lr = actor_lr
        self.optimizer_type = optimizer_type
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # params for epsilon greedy
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.use_cuda = use_cuda and th.cuda.is_available()

        self.actor = ActorNetwork(self.state_dim, self.actor_hidden_size,
                                  self.action_dim, self.actor_output_act)

        self.target = ActorNetwork(self.state_dim, self.actor_hidden_size,
                                   self.action_dim, self.actor_output_act)
        self.update_target()

        if self.optimizer_type == "adam":
            self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
        elif self.optimizer_type == "rmsprop":
            self.actor_optimizer = RMSprop(self.actor.parameters(), lr=self.actor_lr)
        if self.use_cuda:
            self.actor.cuda()
            self.target.cuda()

    # train on a sample batch
    def learn(self):

        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.state_dim)
        actions_var = to_tensor_var(batch.actions, self.use_cuda, "long").view(-1, 1)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, 1)
        next_states_var = to_tensor_var(batch.next_states, self.use_cuda).view(-1, self.state_dim)
        dones_var = to_tensor_var(batch.dones, self.use_cuda).view(-1, 1)

        # compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        current_q = self.actor(states_var).gather(1, actions_var)

        # compute V(s_{t+1}) for all next states and all actions,
        # and we then take max_a { V(s_{t+1}) }
        next_state_action_values = self.target(next_states_var).detach()
        next_q = th.max(next_state_action_values, 1)[0].view(-1, 1)
        # compute target q by: r + gamma * max_a { V(s_{t+1}) }
        target_q = self.reward_scale * rewards_var + self.reward_gamma * next_q * (1. - dones_var)

        # update value network
        self.actor_optimizer.zero_grad()

        loss = th.nn.MSELoss()(current_q, target_q)
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

    def shared_learning(self, n_agents, agent_index, shared_batch_sample):
        self.learn()

    def update_target(self):
        self.target.load_state_dict(self.actor.state_dict())

    # choose an action based on state with random noise added for exploration in training
    def exploration_action(self, state, epsilon=None):
        if epsilon:
            self.epsilon = epsilon
        else:
            self.decay_epsilon()

        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_dim)
        else:
            action = self.action(state)
        return action

    # choose an action based on state for execution
    def action(self, state):
        state_var = to_tensor_var([state], self.use_cuda)
        state_action_value_var = self.actor(state_var)
        if self.use_cuda:
            state_action_value = state_action_value_var.data.cpu().numpy()[0]
        else:
            state_action_value = state_action_value_var.data.numpy()[0]
        action = np.argmax(state_action_value)
        return action

    def remember(self, state, action, reward, new_state, done):
        self.memory.push(state, action, reward, new_state, done)

    def decay_epsilon(self):
        # decrease the exploration rate epsilon over time
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_end, self.epsilon)

    def save(self, global_step, out_dir='/model'):
        actor_file_path = out_dir + f"/actor_.pt"
        th.save({'global_step': global_step,
                 'model_state_dict': self.actor.state_dict(),
                 'optimizer_state_dict': self.actor_optimizer.state_dict()},
                actor_file_path)
