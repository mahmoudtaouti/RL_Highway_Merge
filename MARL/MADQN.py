import os

import numpy as np
import torch.nn as nn
import torch as th

from MARL.agent.DQN import DQN
from MARL.common.Memory import ReplayMemory
from MARL.common.Model import QMIXNet
from MARL.common.utils import exponential_epsilon_decay
from util.ModifiedTensorBoard import ModifiedTensorBoard


class MADQN:
    """
    multi-agent Deep Q-Network
    - training with concurrent or centralized learnings
    - two model option torch or tensor
    - using target model for stable learning
    - exploration action with exponential epsilon greedy
    @mahmoudtaouti
    """

    def __init__(self, state_dim, action_dim, n_agents, memory_capacity=10000,
                 reward_gamma=0.99, reward_scale=1.,
                 actor_hidden_size=128, target_update_freq=50,
                 actor_output_act=nn.functional.tanh, critic_loss="mse",
                 actor_lr=0.001, optimizer_type="adam",
                 max_grad_norm=0.5, batch_size=64, epsilon_start=0.9,
                 epsilon_end=0.01, epsilon_decay=0.003,
                 qmix_hidden_size=128, training_strategy="concurrent",
                 use_cuda=False, model_type="torch", outputs_dir="logs/"):

        assert training_strategy in ["concurrent", "centralized"]
        assert model_type in ["torch", "tensor"]

        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.batch_size = batch_size

        self.training_strategy = training_strategy
        self.use_cuda = use_cuda and th.cuda.is_available()
        self.epsilon = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.model_type = model_type
        self.tensorboard = ModifiedTensorBoard(outputs_dir)
        self.actor_output_act = actor_output_act

        self.agents = []

        self.agents = [DQN(state_dim=state_dim,
                           action_dim=action_dim,
                           memory_capacity=memory_capacity,
                           reward_gamma=reward_gamma,
                           reward_scale=reward_scale,
                           actor_hidden_size=actor_hidden_size,
                           critic_loss=critic_loss,
                           actor_lr=actor_lr,
                           target_update_freq=target_update_freq,
                           optimizer_type=optimizer_type,
                           batch_size=batch_size,
                           epsilon_start=epsilon_start,
                           epsilon_end=epsilon_end,
                           epsilon_decay=epsilon_decay,
                           max_grad_norm=max_grad_norm,
                           use_cuda=use_cuda)] * n_agents

        if self.training_strategy == "centralized":
            # parameters sharing with shared replay memory
            self.shared_memory = ReplayMemory(capacity=memory_capacity)
            self.qmix_net = QMIXNet(n_agents, state_dim, qmix_hidden_size, hyper_hidden_dim, hyper_layers_num)

    def learn(self):
        """
        train for each agent the gathered experiences
        """
        if self.training_strategy == "concurrent":
            for agent in self.agents:
                agent.learn()
        elif self.training_strategy == "centralized":
            batch = self.shared_memory.sample(self.batch_size)
            for agent in self.agents:
                agent.shared_learning(n_agents=self.n_agents, agent_index=agent, shared_batch_sample=batch, qmix_net= self.qmix_net)

    def remember(self, states, actions, rewards, new_states, dones):
        """
        push experiences to replay memory
        """
        dones = dones if isinstance(dones, list) else [dones] * self.n_agents
        if self.training_strategy == 'concurrent':
            for agent, s, a, r, n_s, d in zip(self.agents, states, actions, rewards, new_states, dones):
                agent.remember(s, a, r, n_s, d)
        elif self.training_strategy == 'centralized':
            self.shared_memory.push(states, actions, rewards, new_states, dones)

    def exploration_act(self, states, n_episodes):
        """
        for each agent make exploration action with exponential epsilon greedy,
        and return a tuple of all agents actions
        Returns:
            tuple(actions)
        """
        self.epsilon = exponential_epsilon_decay(
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            decay_rate=self.epsilon_decay,
            episode=n_episodes)

        actions = []
        if self.training_strategy == 'concurrent':
            for agent, state in zip(self.agents, states):
                action = agent.exploration_action(state, epsilon=self.epsilon)
                actions.append(action)
        elif self.training_strategy == 'centralized':
            for agent, _ in zip(self.agents, states):
                action = agent.exploration_action(np.array(states).reshape(-1, self.n_agents * self.state_dim),
                                                  epsilon=self.epsilon)
                actions.append(action)
        return tuple(actions)

    def update_targets(self):
        """
        update target model weights for each agent
        """
        for agent in self.agents:
            agent.update_target()

    def act(self, states):
        """
        for each agent predict action,
        Returns:
            tuple(actions)
        """
        actions = []
        if self.training_strategy == 'concurrent':
            for agent, state in zip(self.agents, states):
                action = agent.action(state)
                actions.append(action)
        elif self.training_strategy == 'centralized':
            for agent, _ in zip(self.agents, states):
                action = agent.action(np.array(states).reshape(-1, self.n_agents * self.state_dim))
                actions.append(action)
        return tuple(actions)

    def save(self, out_dir, checkpoint_num, global_step):
        """
        create checkpoint directory,
        and save models of all RL agents
        """
        os.makedirs(out_dir + "/models", exist_ok=True)
        checkpoint_dir = out_dir + f"/models/checkpoint-{checkpoint_num}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        for index, agent in enumerate(self.agents):
            if self.model_type == "torch":
                actor_file_path = checkpoint_dir + f"/actor_{index}.pt"
                th.save({'global_step': global_step,
                         'model_state_dict': agent.actor.state_dict(),
                         'optimizer_state_dict': agent.actor_optimizer.state_dict()},
                        actor_file_path)
            else:
                actor_file_path = checkpoint_dir + f"/eps{global_step}_actor_{index}.model"
                agent.actor.save(actor_file_path)

    def load(self, directory, check_point=None):
        """
        load saved models
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The directory '{directory}' does not exist.")

        checkpoint_dir = os.path.join(directory, f"checkpoint-{check_point}") if check_point else directory

        for index, agent in enumerate(self.agents):
            actor_file_path = os.path.join(checkpoint_dir, f"actor_{index}.pt")

            checkpoint = th.load(actor_file_path)
            agent.actor.load_state_dict(checkpoint['model_state_dict'])
            agent.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
