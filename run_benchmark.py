import numpy as np

from MARL.MAA2C import MAA2C
from MARL.common.utils import agg_list_stat
from on_ramp_env import OnRampEnv
import argparse
import os
from util.common_util import increment_counter
import config as cnf


def arg_parse():
    # create an argument parser
    parser = argparse.ArgumentParser(description="On-Ramp Highway Merging Scenario runner.")

    # parser.add_argument('-m', "--tensor_model", required=True, type=str,
    #                    help='Define the path to tensorflow model for zipper merge e1.')
    parser.add_argument(
        '--sync-with-carla',
        action='store_true',
        help='run synchronization between SUMO and CARLA')
    parser.add_argument(
        '--show-gui',
        action='store_true',
        help='show SUMO gui')

    parser.add_argument(
        '-S', "--training-strategy",
        required=False, type=str,
        choices=['centralized', 'concurrent'],
        help='Choose training strategy [centralized, concurrent], default=concurrent')

    opt = parser.parse_args()
    return opt


def main():
    # parse the arguments
    opt = arg_parse()

    exec_num = increment_counter()
    outputs_dir = f"./outputs/{exec_num}/"
    os.makedirs(outputs_dir, exist_ok=True)
    # create environment and agent
    env = OnRampEnv(exec_num=exec_num)

    rl = MAA2C(n_agents=env.n_agents, state_dim=env.n_state, action_dim=env.n_action,
               memory_capacity=cnf.MEMORY_SIZE, batch_size=cnf.BATCH_SIZE,
               reward_gamma=cnf.REWARD_DISCOUNTED_GAMMA,
               actor_hidden_size=256, critic_hidden_size=256,
               epsilon_start=cnf.EPSILON_START, epsilon_end=cnf.EPSILON_END,
               epsilon_decay=cnf.EPSILON_DECAY,
               optimizer_type="rmsprop", training_strategy=cnf.TRAINING_STRATEGY)

    eva_num = 0
    for eps in range(cnf.EPISODES):
        states, _ = env.reset(opt.show_gui, opt.sync_with_carla)
        done = False
        step = 0
        rl.tensorboard.step = eps
        while not done:
            step += 1
            # select agents action
            actions = rl.exploration_act(states, n_episodes=eps)

            # perform actions on env
            new_state, global_reward, done, info = env.step(actions)

            # TODO: implement rewards for centralized learning

            # global reward for each agent
            rewards = [locl_r + global_reward for locl_r in info["local_rewards"]]

            # remember experience
            rl.remember(states, actions, rewards, new_state, done)

        if eps > cnf.EPISODES_BEFORE_TRAIN:
            rl.learn()

        env.close()

        if eps != 0 and eps % cnf.EVAL_INTERVAL == 0:
            eva_num += 1
            rewards = []
            speeds = []
            ttcs = []
            headways = []
            infos = []
            trip_time = []
            local_rewards = []
            for i in range(cnf.EVAL_EPISODES):
                rewards_i = []
                infos_i = []
                state, _ = env.reset(show_gui=True)
                eval_done = False
                while not eval_done:
                    action = rl.act(state)
                    state, global_reward, eval_done, info = env.step(action)
                    total_reward = sum(info["local_rewards"]) + global_reward
                    local_rewards.append(info["local_rewards"])
                    if i % 5 == 0:
                        env.render(eva_num) if i == 0 else None
                    rewards_i.append(total_reward)
                    infos_i.append(info)
                    for agent in range(0, env.n_agents):
                        speeds.append(state[agent][2])
                        ttcs.append(state[agent][7])
                        headways.append(state[agent][8])
                        trip_time.append(state[agent][9])
                env.close()
                rewards.append(rewards_i)
                infos.append(infos_i)

            rewards_mu, rewards_std, r_max, r_min = agg_list_stat(rewards)
            speeds = np.array(speeds)
            ttcs = np.array(ttcs)
            headways = np.array(headways)
            local_rewards = np.array(local_rewards).reshape(-1, 2)
            locl_r_sum = np.sum(local_rewards, axis=0)
            trip_time = np.array(trip_time)
            rl.tensorboard.update_stats(
                reward_avg=rewards_mu,
                veh0_locl_reward=locl_r_sum[0],
                veh1_locl_reward=locl_r_sum[1],
                reward_std=rewards_std,
                epsilon=rl.epsilon,
                speed_avg=np.mean(speeds),
                min_ttc=np.min(ttcs),
                headways=np.min(headways),
                trip_time=np.max(trip_time))

            rl.save(out_dir=outputs_dir, checkpoint_num=eva_num, global_step=eps)

# class Runner:
#     def __init__(self, rl_option, training_strategy):
#
#         self.exec_num = increment_counter()
#         self.rl_option = rl_option
#         self.training_strategy = training_strategy
#         self.roll_out_steps = cnf.ROLL_OUT_STEPS
#         self.episode_done = False
#         self.env = OnRampEnv(exec_num=self.exec_num)
#
#         self.outputs_dir = f"./outputs/{self.exec_num}/"
#         os.makedirs(self.outputs_dir, exist_ok=True)
#
#         self.n_episodes = 0
#         self.n_eval_episodes = 0
#
#         if rl_option == "MADQN":
#             self.rl = MADQN(self.env, memory_capacity=cnf.MEMORY_SIZE,
#                             state_dim=self.env.n_state, action_dim=self.env.n_action,
#                             batch_size=cnf.BATCH_SIZE,
#                             reward_gamma=cnf.REWARD_DISCOUNTED_GAMMA,
#                             actor_hidden_size=256, critic_hidden_size=256,
#                             epsilon_start=cnf.EPSILON_START, epsilon_end=cnf.EPSILON_END,
#                             epsilon_decay=cnf.EPSILON_DECAY, use_cuda=False,
#                             reward_type="global_R", target_update_freq=cnf.UPDATE_TARGET_FREQ,
#                             optimizer_type="adam")
#         elif rl_option == "MAA2C":
#             self.rl = MAA2C(n_agents=self.env.n_agents, state_dim=self.env.n_state, action_dim=self.env.n_action,
#                             memory_capacity=cnf.MEMORY_SIZE, batch_size=cnf.BATCH_SIZE,
#                             reward_gamma=cnf.REWARD_DISCOUNTED_GAMMA,
#                             actor_hidden_size=256, critic_hidden_size=256,
#                             epsilon_start=cnf.EPSILON_START, epsilon_end=cnf.EPSILON_END,
#                             epsilon_decay=cnf.EPSILON_DECAY,
#                             optimizer_type="rmsprop", training_strategy=training_strategy)
#         else:
#             raise ValueError(f"no rl_option for {rl_option}")
#
#     def train_n_steps(self):
#         done = False
#
#         if self.episode_done:
#             env_state, _ = self.env.reset()
#             n_steps = 0
#
#         states = []
#         actions = []
#         rewards = []
#         # take n steps
#         for _ in range(self.roll_out_steps):
#             states.append(self.env.state)
#             actions = self.rl.exploration_act(self.env.state, self.n_episodes)
#             next_state, reward, done, _ = self.env.step(actions)
#             actions.append(actions)
#             rewards.append(reward)
#             final_state = next_state
#             if done:
#                 break
#
#         # discount reward
#         if done:
#             final_r = [0.0] * self.env.n_agents
#             self.n_episodes += 1
#             self.rl.tensorboard.step = self.n_episodes
#             self.episode_done = True
#         else:
#             self.episode_done = False
#             final_action = self.rl.act(final_state)
#             one_hot_action = [index_to_one_hot(a, self.env.n_action) for a in final_action]
#             final_r = self.rl.value(final_state, one_hot_action)
#
#
#     def evaluation(self):
#
#         for i in range(cnf.EVAL_EPISODES):
#
#             state, _ = self.env.reset(show_gui=True)
#             eval_done = False
#             while not eval_done:
#                 action = self.rl.act(state)
#                 state, global_reward, eval_done, info = self.env.step(action)
#
#                 if i % 5 == 0:
#                     self.env.render(i) if i == 0 else None
#
#             self.env.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
