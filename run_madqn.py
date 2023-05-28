import numpy as np
import os

from MARL.MADQN import MADQN
from MARL.common.utils import agg_list_stat
from on_ramp_env import OnRampEnv
from util.common_util import increment_counter

import config as cnf


def main():
    """
    start point for training with MADQN
    Scenario: On-Ramp Merging
    """
    exec_num = increment_counter()
    outputs_dir = f"./outputs/{exec_num}/"
    os.makedirs(outputs_dir, exist_ok=True)
    # create environment and agent
    env = OnRampEnv(exec_num=exec_num)

    rl = MADQN(n_agents=env.n_agents, state_dim=env.n_state, action_dim=env.n_action,
               memory_capacity=cnf.MEMORY_SIZE, batch_size=cnf.BATCH_SIZE,
               target_update_freq=50, reward_gamma=cnf.REWARD_DISCOUNTED_GAMMA, actor_hidden_size=256,
               epsilon_start=cnf.EPSILON_START, epsilon_end=cnf.EPSILON_END, epsilon_decay=cnf.EPSILON_DECAY,
               optimizer_type="rmsprop", training_strategy=cnf.TRAINING_STRATEGY, model_type=cnf.MODEL_TYPE)

    training_loop(env, rl, outputs_dir)


def training_loop(env, rl, outputs_dir):
    """
    training loop
    while not done get exploration action,
    decay epsilon, do env step, push new_sates to replay memory
    train after some episodes, update target model weights
    """
    eva_num = 0
    for eps in range(cnf.EPISODES):
        states, _ = env.reset()
        done = False
        step = 0
        rl.tensorboard.step = eps
        while not done:
            step += 1
            # select agents action
            actions = rl.exploration_act(states, n_episodes=eps)

            # perform actions on env
            new_states, global_reward, done, info = env.step(actions)

            # global reward for each agent
            rewards = [locl_r + global_reward for locl_r in info["local_rewards"]]

            # remember experience
            rl.remember(states, actions, rewards, new_states, done)
            states = new_states

        if eps > cnf.EPISODES_BEFORE_TRAIN:
            rl.learn()

        if eps % cnf.UPDATE_TARGET_FREQ == 0:
            rl.update_targets()

        env.close()

        if eps != 0 and eps % cnf.EVAL_INTERVAL == 0:
            evaluation(env, rl, eps, eva_num, outputs_dir)
            eva_num += 1


def evaluation(env, rl, episode, eva_number, outputs_dir):
    """
    # start with sumo gui
    # render images to outputs_dir
    # save tensorboard logs
    # save checkpoint model
    """
    rewards = []
    speeds = []
    ttcs = []
    headways = []
    infos = []
    trip_time_delays = []
    local_rewards = []
    for i in range(cnf.EVAL_EPISODES):
        rewards_i = []
        infos_i = []
        states, _ = env.reset(show_gui=True)
        eval_done = False
        in_step = 0
        while not eval_done:
            in_step += 1
            action = rl.act(states)
            new_states, global_reward, eval_done, info = env.step(action)
            total_reward = sum(info["local_rewards"]) + global_reward
            local_rewards.append(info["local_rewards"])
            if in_step % 5 == 0:
                env.render(eva_number) if i == 0 else None
            rewards_i.append(total_reward)
            infos_i.append(info)
            for agent in range(0, env.n_agents):
                speeds.append(states[agent][2])
                ttcs.append(states[agent][7])
                headways.append(states[agent][8])
                trip_time_delays.append(states[agent][9])

            states = new_states
        env.close()
        rewards.append(rewards_i)
        infos.append(infos_i)

    rewards_mu, rewards_std, r_max, r_min = agg_list_stat(rewards)
    speeds = np.array(speeds)
    ttcs = np.array(ttcs)
    headways = np.array(headways)
    local_rewards = np.array(local_rewards).reshape(-1, 2)
    locl_r_sum = np.sum(local_rewards, axis=0)
    trip_time_delays = np.array(trip_time_delays)
    rl.tensorboard.update_stats(
        reward_avg=rewards_mu,
        veh0_locl_reward=locl_r_sum[0],
        veh1_locl_reward=locl_r_sum[1],
        reward_std=rewards_std,
        epsilon=rl.epsilon,
        speed_avg=np.mean(speeds),
        min_ttc=np.min(ttcs),
        headway=np.min(headways),
        trip_time=np.max(trip_time_delays))

    rl.save(out_dir=outputs_dir, checkpoint_num=eva_number, global_step=episode)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
