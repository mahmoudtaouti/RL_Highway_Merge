import os

from MARL.agent.DDPG import DDPG
import torch.nn as nn


class MADDPG:
    """
    A multi-agent learned with DDPG
    reference: https://github.com/ChenglongChen/pytorch-DRL
    """
    def __init__(self, state_dim, action_dim, n_agents, memory_capacity=10000,
                 target_tau=0.01, target_update_steps=5, reward_gamma=0.99, reward_scale=1.,
                 actor_hidden_size=128, critic_hidden_size=128,
                 actor_output_act=nn.functional.tanh, critic_loss="mse",
                 actor_lr=0.001, critic_lr=0.001, optimizer_type="adam", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, epsilon_start=0.9,
                 epsilon_end=0.01, epsilon_decay=200, use_cuda=False):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents

        # Create N agents
        self.agents = [DDPG(state_dim=state_dim,
                            action_dim=action_dim,
                            memory_capacity=memory_capacity,
                            reward_gamma=reward_gamma,
                            reward_scale=reward_scale,
                            actor_hidden_size=actor_hidden_size,
                            critic_hidden_size=critic_hidden_size,
                            critic_loss=critic_loss,
                            actor_lr=actor_lr,
                            critic_lr=critic_lr,
                            target_tau=target_tau,
                            target_update_steps=target_update_steps,
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
            agent.learn()

    def remember(self, states, actions, rewards, new_states, dones):
        """
        push experiences to replay memory
        """
        for agent, s, a, r, n_s, d in zip(self.agents, states, actions, rewards, new_states, dones):
            agent.remember(s, a, r, n_s, d)

    def exploration_act(self, states):
        """
        for each agent make exploration action,
        and return a tuple of all agents actions
        """
        actions = []
        for agent, state in zip(self.agents, states):
            action = agent.exploration_action(state)
            actions.append(action)
        return tuple(actions)

    def act(self, states):
        """
        for each agent predict action,
        and return a tuple of all agents actions
        """
        actions = []
        for agent, state in zip(self.agents, states):
            action = agent.act(state)
            actions.append(action)
        return tuple(actions)

    def save(self, out_dir, checkpoint_num):
        """
        save models of all RL agents
        """
        checkpoint_dir = out_dir + f"/checkpoint-{checkpoint_num}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        for agent in self.agents:
            agent.save(checkpoint_dir)