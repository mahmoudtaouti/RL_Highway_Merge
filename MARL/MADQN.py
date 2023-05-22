import configparser
import logging
import os

import numpy as np
import torch as th
from torch import nn
from torch.optim import Adam, RMSprop

from MARL.common.Memory import ReplayMemory
from MARL.common.Model import ActorNetwork
from MARL.common.utils import agg_double_list, exponential_epsilon_decay, identity, to_tensor_var
from util.ModifiedTensorBoard import ModifiedTensorBoard

config_dir = 'configs/configs_dqn.ini'
config = configparser.ConfigParser()
config.read(config_dir)
torch_seed = 0  # CLS config.getint('MODEL_CONFIG', 'torch_seed')
th.manual_seed(torch_seed)
th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True


class MADQN:
    """
    A multi-agent learned with DQN
    reference: https://github.com/ChenglongChen/pytorch-DRL
    """

    def __init__(self, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=10000,
                 reward_gamma=0.99, reward_scale=20.,
                 actor_hidden_size=128, critic_hidden_size=128,
                 actor_output_act=identity, critic_loss="mse",
                 actor_lr=0.001, critic_lr=0.001,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=0.001,
                 use_cuda=False, target_update_freq=4, reward_type="regionalR", outputs_dir="outputs/"):

        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_episodes = 0
        self.n_steps = 0
        self.max_steps = max_steps
        self.tensorboard = ModifiedTensorBoard()
        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale
        self.episode_done = True
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
        self.n_agents = len(self.env.controlled_vehicles)

        # params for epsilon greedy
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.use_cuda = use_cuda and th.cuda.is_available()
        self.q_network = ActorNetwork(self.state_dim, self.actor_hidden_size,
                                      self.action_dim, self.actor_output_act)
        self.target_network = ActorNetwork(self.state_dim, self.actor_hidden_size,
                                           self.action_dim, self.actor_output_act)
        self.target_update_freq = target_update_freq
        self.reward_type = reward_type
        self.episode_rewards = [0]

        if self.optimizer_type == "adam":
            self.q_network_optimizer = Adam(self.q_network.parameters(), lr=self.actor_lr)
        elif self.optimizer_type == "rmsprop":
            self.q_network_optimizer = RMSprop(self.q_network.parameters(), lr=self.actor_lr)
        if self.use_cuda:
            self.q_network.cuda()
            self.target_network.cuda()

    # agent interact with the environment to collect experience
    def interact(self):

        if self.episode_done:
            self.env_state, _ = self.env.reset()
            self.n_steps = 0

        state = self.env_state
        action = self.exploration_action(self.env_state)
        next_state, global_reward, done, info = self.env.step(tuple(action))

        self.episode_rewards[-1] += global_reward

        if self.reward_type == "regionalR":
            reward = list(info["regional_rewards"])
        elif self.reward_type == "global_R":
            reward = [global_reward] * self.n_agents

        if done:
            self.n_episodes += 1
            self.tensorboard.step = self.n_episodes
            self.episode_done = True
            self.episode_rewards.append(0)
        else:
            self.env_state = next_state
            self.episode_done = False

        self.n_steps += 1
        for agent_id in range(self.n_agents):
            self.memory.push(state[agent_id], action[agent_id], reward[agent_id], next_state[agent_id], done)

    def remember(self, state, actions, rewards, new_state, done):
        for agent_id in range(self.n_agents):
            self.memory.push(state[agent_id], actions[agent_id], rewards[agent_id], new_state[agent_id], done)

    # train on a sample batch
    def train(self):
        if self.n_episodes <= self.episodes_before_train:
            pass
        # update 10 times
        for _ in range(10):
            batch = self.memory.sample(self.batch_size)
            states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.state_dim)
            actions_var = to_tensor_var(batch.actions, self.use_cuda, "long").view(-1, 1)
            rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, 1)
            next_states_var = to_tensor_var(batch.next_states, self.use_cuda).view(-1, self.state_dim)
            dones_var = to_tensor_var(batch.dones, self.use_cuda).view(-1, 1)

            # compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
            current_q = self.q_network(states_var).gather(1, actions_var)

            # compute V(s_{t+1}) for all next states and all actions,
            # and we then take max_a { V(s_{t+1}) }
            next_state_action_values = self.target_network(next_states_var).detach()
            next_q = th.max(next_state_action_values, 1)[0].view(-1, 1)
            # compute target q by: r + gamma * max_a { V(s_{t+1}) }
            target_q = self.reward_scale * rewards_var + self.reward_gamma * next_q * (1. - dones_var)

            # update value network
            self.q_network_optimizer.zero_grad()
            if self.critic_loss == "huber":
                loss = th.nn.functional.smooth_l1_loss(current_q, target_q)
            else:
                loss = th.nn.MSELoss()(current_q, target_q)
            loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm(self.q_network.parameters(), self.max_grad_norm)
            self.q_network_optimizer.step()

        # Periodically update the target network by Q network to target Q network
        if self.n_episodes % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    # choose an action based on state with random noise added for exploration in training
    def exploration_action(self, state):
        self.epsilon = exponential_epsilon_decay(
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            decay_rate=self.epsilon_decay,
            episode=self.n_episodes)
        actions = [0] * self.n_agents
        for agent_id in range(self.n_agents):
            if np.random.rand() < self.epsilon:
                actions[agent_id] = np.random.choice(self.action_dim)
            else:
                actions = self.action(state)
        return actions

    # choose an action based on state for execution
    def action(self, state):
        state_var = to_tensor_var([state], self.use_cuda)
        state_action_value_var = self.q_network(state_var)
        if self.use_cuda:
            state_action_value = state_action_value_var.data.cpu().numpy()[0]
        else:
            state_action_value = state_action_value_var.data.numpy()[0]

        action = np.argmax(state_action_value, axis=1)
        return action

    # evaluation the learned agents
    def evaluation(self, eval_episodes=10):

        rewards = []
        speeds = []
        ttcs = []
        headways = []
        infos = []

        for i in range(eval_episodes):
            rewards_i = []
            infos_i = []
            state, _ = self.env.reset(show_gui=True)
            action = self.action(state)
            state, reward, done, info = self.env.step(action)

            rewards_i.append(reward)
            infos_i.append(info)
            step = 0
            while not done:
                step += 1
                action = self.action(state)
                state, reward, done, info = self.env.step(action)
                # if(step % 5 == 0):
                #    self.env.render(eval_num) if i == 0 else None
                rewards_i.append(reward)
                infos_i.append(info)

                for agent in range(0, self.n_agents):
                    speeds.append(state[agent][2])
                    ttcs.append(state[agent][7])
                    headways.append(state[agent][8])
            rewards.append(rewards_i)
            infos.append(infos_i)

        rewards_mu, rewards_std, r_max, r_min = agg_double_list(rewards)

        self.tensorboard.update_stats(
            reward_avg=rewards_mu,
            reward_std=rewards_std,
            epsilon=self.epsilon,
            avg_speed=np.mean(speeds),
            min_ttc=np.min(ttcs),
            headway=np.min(headways))

        return rewards, infos

    # discount roll out rewards
    def _discount_reward(self, rewards, final_value):
        discounted_r = np.zeros_like(rewards)
        running_add = final_value
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.reward_gamma + rewards[t]
            discounted_r[t] = running_add
        return discounted_r

    def load(self, model_dir, global_step=None, train_mode=False):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if global_step is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        tokens = file.split('.')[0].split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = file
                            save_step = cur_step
            else:
                save_file = 'checkpoint-{:d}.pt'.format(global_step)
        if save_file is not None:
            file_path = model_dir + '/' + save_file
            checkpoint = th.load(file_path)
            print('Checkpoint loaded: {}'.format(file_path))
            # logging.info('Checkpoint loaded: {}'.format(file_path))
            self.q_network.load_state_dict(checkpoint['model_state_dict'])
            if train_mode:
                self.q_network_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.q_network.train()
            else:
                self.q_network.eval()
            return True
        logging.error('Can not find checkpoint for {}'.format(model_dir))
        return False

    def save(self, model_dir, global_step):
        os.makedirs(model_dir + "models", exist_ok=True)
        file_path = model_dir + 'models/checkpoint-{:d}.pt'.format(global_step)
        th.save({'global_step': global_step,
                 'model_state_dict': self.q_network.state_dict(),
                 'optimizer_state_dict': self.q_network_optimizer.state_dict()},
                file_path)
