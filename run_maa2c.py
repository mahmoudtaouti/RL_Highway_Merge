import numpy as np
import os

from MARL.MAA2C import MAA2C
from MARL.common.utils import agg_list_stat
from on_ramp_env import OnRampEnv
from util.common_util import increment_counter, write_to_log

import config as cnf


def main():
    """
    start point for training with MAA2C
    Scenario: On-Ramp Merging
    """
    exec_num = increment_counter()
    outputs_dir = f"./outputs/{exec_num}/"
    os.makedirs(outputs_dir, exist_ok=True)

    write_to_log(f"EXEC==================================================\n"
                 f"Execution number : {exec_num}\n"
                 f"RL option: MAA2C \n"
                 f"Training strategy: {cnf.TRAINING_STRATEGY} \n"
                 "=======================================================", output_dir=outputs_dir)

    env = OnRampEnv(exec_num=exec_num)

    rl = MAA2C(n_agents=env.n_agents, state_dim=env.n_state, action_dim=env.n_action,
               memory_capacity=cnf.MEMORY_SIZE, batch_size=cnf.BATCH_SIZE,
               reward_gamma=cnf.REWARD_DISCOUNTED_GAMMA,
               actor_hidden_size=256, critic_hidden_size=256,
               epsilon_start=cnf.EPSILON_START, epsilon_end=cnf.EPSILON_END,
               epsilon_decay=cnf.EPSILON_DECAY,
               optimizer_type="rmsprop", training_strategy=cnf.TRAINING_STRATEGY, outputs_dir=outputs_dir)

    rl.load(directory="./outputs/17/models", check_point=7)

    training_loop(env, rl, outputs_dir)


def training_loop(env, rl, outputs_dir):
    """
    training loop
    while not done get exploration action,
    decay epsilon, do env step, push new_sates to replay memory
    train after some episodes
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
            new_state, global_reward, done, info = env.step(actions)

            # global reward for each agent
            rewards = [locl_r + global_reward for locl_r in info["local_rewards"]]

            # remember experience
            rl.remember(states, actions, rewards, new_state, done)

        if eps > cnf.EPISODES_BEFORE_TRAIN:
            rl.learn()

        env.close()

        if eps != 0 and eps % cnf.EVAL_INTERVAL == 0:
            evaluation(env, rl, eps, eva_num, outputs_dir)
            eva_num += 1


def evaluation(env, rl, episode, eval_number, outputs_dir):
    """
    # start with sumo gui
    # render images to outputs_dir
    # save tensorboard logs
    # save checkpoint model
    # save data to log file
    """
    rewards = []
    speeds = []
    ttcs = []
    headways = []
    infos = []
    trip_time = []
    local_rewards = []
    for i in range(cnf.EVAL_EPISODES):
        write_to_log(f"Evaluation___________________________________________\n"
                     f" number -- {eval_number}\n"
                     f" training episode -- {episode}\n", output_dir=outputs_dir)
        rewards_i = []
        infos_i = []
        state, _ = env.reset(show_gui=True)
        eval_done = False
        while not eval_done:
            action = rl.act(state)
            state, global_reward, eval_done, info = env.step(action)
            total_reward = sum(info["local_rewards"]) + global_reward
            local_rewards.append(info["local_rewards"])
            env.render(eval_number) if i == 0 else None
            rewards_i.append(total_reward)
            infos_i.append(info)
            for agent in range(0, env.n_agents):
                speeds.append(state[agent][2])
                ttcs.append(state[agent][7])
                headways.append(state[agent][8])
                trip_time.append(state[agent][9])
            write_to_log(f"Step---------------------------------\n"
                         f"\t* actions : {action}\n"
                         f"\t* agents dones : {info['agents_dones']}\n"
                         f"\t* state : {state} \n"
                         f"\t* global reward : {global_reward} \n"
                         f"\t* local rewards : {info['local_rewards']} \n", output_dir=outputs_dir)
        write_to_log(f"_____________________________________________\n", output_dir=outputs_dir)
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
        headway=np.min(headways),
        trip_time=np.max(trip_time))

    rl.save(out_dir=outputs_dir, checkpoint_num=eval_number, global_step=episode)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
