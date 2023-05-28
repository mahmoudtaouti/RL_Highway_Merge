import os

from MARL.agent.A2C import A2C
import torch.nn as nn
import torch as th
from torch.optim import Adam

from MARL.common.Memory import ReplayMemory
from MARL.common.Model import CriticNetwork
from MARL.common.utils import exponential_epsilon_decay
from util.ModifiedTensorBoard import ModifiedTensorBoard


class MAA2C:
    """
    multi agent advantage actor critic
    @mahmoudtaouti
    """

    def __init__(self, state_dim, action_dim, n_agents, memory_capacity=10000,
                 reward_gamma=0.99, reward_scale=1.,
                 actor_hidden_size=128, critic_hidden_size=128,
                 actor_output_act=nn.functional.softmax, critic_loss="mse",
                 actor_lr=0.001, critic_lr=0.001, optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=64, epsilon_start=0.9,
                 epsilon_end=0.01, epsilon_decay=0.003,
                 training_strategy="concurrent", use_cuda=False, outputs_dir="logs/"):

        assert training_strategy in ["concurrent", "centralized"]

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

        self.tensorboard = ModifiedTensorBoard(outputs_dir)

        self.shared_critic = None
        self.shared_memory = None
        self.shared_critic_optimizer = None

        if training_strategy == "centralized":
            self.shared_memory = ReplayMemory(capacity=memory_capacity)
            self.shared_critic = CriticNetwork(state_dim * n_agents, action_dim * n_agents, critic_hidden_size, 1)
            self.shared_critic_optimizer = Adam(self.shared_critic.parameters(), lr=critic_lr)

        # Create N agents
        self.agents = []
        for i in range(self.n_agents):
            agent = A2C(state_dim=state_dim,
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
                        use_cuda=use_cuda)
            self.agents.append(agent)

    def learn(self):
        """
        train for each agent the gathered experiences
        agent.shared_learning: centralized learning strategy that share the same critic network
        agent.learn: concurrent (independent) learning
        """
        for index, agent in enumerate(self.agents):
            if self.training_strategy == "centralized":
                shared_batch = self.shared_memory.sample(self.batch_size)
                agent.shared_learning(n_agents=self.n_agents,
                                      agent_index=index,
                                      shared_batch_sample=shared_batch,
                                      shared_critic=self.shared_critic,
                                      shared_critic_optim=self.shared_critic_optimizer)
            else:
                agent.learn()

    def remember(self, states, actions, rewards, new_states, dones):
        """
        push experiences to replay memory
        """
        dones = dones if isinstance(dones, list) else [dones] * self.n_agents
        if self.training_strategy == "centralized":
            self.shared_memory.push(states, actions, rewards, new_states, dones)
        else:
            for agent, s, a, r, n_s, d in zip(self.agents, states, actions, rewards, new_states, dones):
                agent.remember(s, a, r, n_s, d)

    def exploration_act(self, states, n_episodes):
        """
        for each agent make exploration action,
        and return a tuple of all agents actions
        using exponential epsilon decay
        Returns: tuple(actions)
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
        for each agent predict action
        and return a tuple of all agents actions
        """
        actions = []
        for agent, state in zip(self.agents, states):
            action = agent.action(state)
            actions.append(action)
        return tuple(actions)

    def save(self, out_dir, checkpoint_num, global_step):
        """
        save models of all MAA2C agents
        Args:
            out_dir (str): Directory path where to save.
            checkpoint_num(int): check-point during the training
            global_step (int): Global step or checkpoint number to load (optional).

        Raises:
            FileNotFoundError: If the specified output directory does not exist.
        """
        if not os.path.exists(out_dir):
            raise FileNotFoundError(f"The directory '{out_dir}' does not exist.")

        os.makedirs(out_dir + "/models", exist_ok=True)
        checkpoint_dir = out_dir + f"/models/checkpoint-{checkpoint_num}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        for index, agent in enumerate(self.agents):
            actor_file_path = checkpoint_dir + f"/actor_{index}.pt"
            critic_file_path = checkpoint_dir + f"/critic_{index}.pt"
            shared_critic_oath = checkpoint_dir + f"/shared_critic.pt"

            th.save({'global_step': global_step,
                     'model_state_dict': agent.actor.state_dict(),
                     'optimizer_state_dict': agent.actor_optimizer.state_dict()},
                    actor_file_path)

            if self.training_strategy == "centralized":
                th.save({'global_step': global_step,
                         'model_state_dict': self.shared_critic.state_dict(),
                         'optimizer_state_dict': self.shared_critic_optimizer.state_dict()},
                        shared_critic_oath)
            else:
                th.save({'global_step': global_step,
                         'model_state_dict': agent.critic.state_dict(),
                         'optimizer_state_dict': agent.critic_optimizer.state_dict()},
                        critic_file_path)

    def load(self, directory, check_point=None):
        """
        Load saved models
        Args:
            directory (str): Directory path where the saved models are located.
            check_point (int): Global step or checkpoint number to load (optional).
        Raises:
            FileNotFoundError: If the specified directory or checkpoint does not exist.
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The directory '{directory}' does not exist.")

        checkpoint_dir = os.path.join(directory, f"checkpoint-{check_point}") if check_point else directory

        for index, agent in enumerate(self.agents):
            actor_file_path = os.path.join(checkpoint_dir, f"actor_{index}.pt")
            critic_file_path = os.path.join(checkpoint_dir, f"critic_{index}.pt")

            if not os.path.exists(actor_file_path):
                raise FileNotFoundError(f"The actor model file '{actor_file_path}' does not exist.")

            if not os.path.exists(critic_file_path):
                raise FileNotFoundError(f"The critic model file '{critic_file_path}' does not exist.")

            checkpoint = th.load(actor_file_path)
            agent.actor.load_state_dict(checkpoint['model_state_dict'])
            agent.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # TODO : correct load for centralized case
            if self.training_strategy == "centralized":
                critic_checkpoint = th.load(critic_file_path)
                self.shared_critic.load_state_dict(critic_checkpoint['model_state_dict'])
                self.shared_critic_optimizer.load_state_dict(critic_checkpoint['optimizer_state_dict'])
            else:
                critic_checkpoint = th.load(critic_file_path)
                agent.critic.load_state_dict(critic_checkpoint['model_state_dict'])
                agent.critic_optimizer.load_state_dict(critic_checkpoint['optimizer_state_dict'])

        print("Models loaded successfully.")
