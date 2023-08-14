import torch as th
import os
from MARL.common.Model import QNetwork, QMIXNet
from MARL.common.utils import to_tensor_var


class QMIX_Agent:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape = self.obs_shape

        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

        self.eval_q = QNetwork(input_shape, )
        self.target_q = QNetwork(input_shape, )

        self.eval_qmix_net = QMIXNet( N, state_dim, hidden_size, hyper_hidden_dim, hyper_layers_num)

        self.target_q.load_state_dict(self.eval_q.state_dict())

        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_q.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)


    def shared_learning(self, batch, n_agents, qmix_net):

        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.state_dim)
        actions_var = to_tensor_var(batch.actions, self.use_cuda, "long").view(-1, 1)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, 1)
        next_states_var = to_tensor_var(batch.next_states, self.use_cuda).view(-1, self.state_dim)
        dones_var = to_tensor_var(batch.dones, self.use_cuda).view(-1, 1)

        # Compute Q(s_t, a) using the actor network
        current_q = self.actor(states_var).gather(1, actions_var)

        # Compute Q_tot using the mixing network
        q_tot = self.mixing_network.compute_Q_tot(batch.states, batch.actions, self.actor, self.target)

        # Compute the target Q values
        next_state_action_values = self.target(next_states_var).detach()
        next_q = th.max(next_state_action_values, 1)[0].view(-1, 1)
        target_q = self.reward_scale * rewards_var + self.reward_gamma * next_q * (1. - dones_var)

        # Update actor network
        self.actor_optimizer.zero_grad()
        loss = th.nn.functional.smooth_l1_loss(current_q, q_tot)  # Using QMIX loss
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        self.train_count += 1
        if self.train_count % self.target_update_freq == 0:
            self.update_target()


    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_qmix_net.state_dict(), self.model_dir + '/' + num + '_qmix_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')