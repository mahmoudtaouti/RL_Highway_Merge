import os

from MARL.agent.A2C import A2C
import torch.nn as nn
from MARL.common.utils import exponential_epsilon_decay
from util.ModifiedTensorBoard import ModifiedTensorBoard
import torch as th


class MAA2C:

    def __init__(self, state_dim, action_dim, n_agents, memory_capacity=10000,
                 reward_gamma=0.99, reward_scale=1.,
                 actor_hidden_size=128, critic_hidden_size=128,
                 actor_output_act=nn.functional.tanh, critic_loss="mse",
                 actor_lr=0.001, critic_lr=0.001, optimizer_type="adam", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=64, epsilon_start=0.9,
                 epsilon_end=0.01, epsilon_decay=0.003,
                 training_strategy="concurrent", use_cuda=False):

        assert training_strategy in ["concurrent", "centralized"]
        # TODO: implement training based on training_strategy

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.epsilon = epsilon_start

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.tensorboard = ModifiedTensorBoard()

        # Create N agents
        self.agents = [A2C(state_dim=state_dim,
                           action_dim=action_dim,
                           memory_capacity=memory_capacity,
                           reward_gamma=reward_gamma,
                           reward_scale=reward_scale,
                           actor_hidden_size=actor_hidden_size,
                           critic_hidden_size=critic_hidden_size,
                           critic_loss=critic_loss,
                           actor_lr=actor_lr,
                           critic_lr=critic_lr,
                           optimizer_type=optimizer_type,
                           batch_size=batch_size,
                           epsilon_start=epsilon_start,
                           epsilon_end=epsilon_end,
                           epsilon_decay=epsilon_decay,
                           entropy_reg=entropy_reg,
                           actor_output_act=actor_output_act,
                           max_grad_norm=max_grad_norm,
                           use_cuda=use_cuda)] * self.n_agents

    def learn(self):
        """
        train for each agent the gathered experiences in the replay memory,
        """
        for agent in self.agents:
            agent.train()

    def remember(self, states, actions, rewards, new_states, dones):
        """
        push experiences to replay memory
        """
        # TODO: remember based on training strategy
        dones = dones if isinstance(dones, list) else [dones] * self.n_agents
        for agent, s, a, r, n_s, d in zip(self.agents, states, actions, rewards, new_states, dones):
            agent.remember(s, a, r, n_s, d)

    def exploration_act(self, states, n_episodes):
        """
        for each agent make exploration action,
        and return a tuple of all agents actions
        """
        self.epsilon = exponential_epsilon_decay(
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            decay_rate=self.epsilon_decay,
            episode=n_episodes)

        actions = []
        for agent, state in zip(self.agents, states):
            action = agent.exploration_action(state, epsilon=self.epsilon)
            actions.append(action)
        return tuple(actions)

    def act(self, states):
        """
        for each agent predict action,
        and return a tuple of all agents actions
        """
        actions = []
        for agent, state in zip(self.agents, states):
            action = agent.action(state)
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
            actor_file_path = checkpoint_dir + f"/actor_{index}.pt"
            critic_file_path = checkpoint_dir + f"/critic_{index}.pt"

            th.save({'global_step': global_step,
                     'model_state_dict': agent.state_dict(),
                     'optimizer_state_dict': agent.actor_optimizer.state_dict()},
                    actor_file_path)
            th.save({'global_step': global_step,
                     'model_state_dict': agent.critic.state_dict(),
                     'optimizer_state_dict': agent.critic_optimizer.state_dict()},
                    critic_file_path)

    def load(self, directory, global_step=None):
        pass
