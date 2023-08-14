import argparse
import os

import numpy as np

import MADQN_config
import MAA2C_config
from MARL.MAA2C import MAA2C
from MARL.MADQN import MADQN
from MARL.common.utils import agg_stat_list
from on_ramp_env import OnRampEnv
from util.common_util import write_to_log, to_ndarray, INFINITY


def arg_parse():
    # create an argument parser
    parser = argparse.ArgumentParser(description="Zipper merge e1 runner.")

    parser.add_argument('-m', "--model-dir", required=False, type=str,
                        help='Define the path to model.')
    parser.add_argument('-O', "--rl-option", required=False, type=str, choices=["MADQN", "MAA2C"],
                        help='Define reinforcement learning option ["MADQN", "MAA2C"].')
    parser.add_argument('-S', "--training-strategy", required=False, type=str, choices=["concurrent", "centralized"],
                        help='Define the training strategy ["concurrent", "centralized"].')
    parser.add_argument(
        '--sync-with-carla',
        action='store_true',
        help='run synchronization between SUMO and CARLA')
    parser.add_argument(
        '--show-gui',
        action='store_true',
        help='show SUMO gui')
    opt = parser.parse_args()
    return opt


def main():
    opt = arg_parse()

    rl_option = "MAA2C"
    check_point = 99
    outputs_dir = "./outputs/107"
    eval_episodes = 100

    test_dir = os.path.join(outputs_dir, "benchmark")
    os.makedirs(test_dir, exist_ok=True)

    env = OnRampEnv(exec_num=check_point)

    if rl_option == "MAA2C":
        rl = MAA2C(n_agents=env.n_agents, state_dim=env.n_state, action_dim=env.n_action,
                   memory_capacity=MAA2C_config.MEMORY_SIZE, batch_size=MAA2C_config.BATCH_SIZE,
                   reward_gamma=MAA2C_config.REWARD_DISCOUNTED_GAMMA,
                   actor_hidden_size=MAA2C_config.ACTOR_HIDDEN_SIZE, critic_hidden_size=MAA2C_config.CRITIC_HIDDEN_SIZE,
                   epsilon_start=MAA2C_config.EPSILON_START, epsilon_end=MAA2C_config.EPSILON_END,
                   epsilon_decay=MAA2C_config.EPSILON_DECAY,
                   optimizer_type=MAA2C_config.OPTIMIZER_TYPE, training_strategy=MAA2C_config.TRAINING_STRATEGY,
                   is_evaluation=True, outputs_dir=test_dir)
    elif rl_option == "MADQN":
        rl = MADQN(n_agents=env.n_agents, state_dim=env.n_state, action_dim=env.n_action,
                   memory_capacity=MADQN_config.MEMORY_SIZE, batch_size=MADQN_config.BATCH_SIZE,
                   target_update_freq=50, reward_gamma=MADQN_config.REWARD_DISCOUNTED_GAMMA,
                   actor_hidden_size=MADQN_config.ACTOR_HIDDEN_SIZE, critic_loss=MADQN_config.CRITIC_LOSS,
                   epsilon_start=MADQN_config.EPSILON_START, epsilon_end=MADQN_config.EPSILON_END,
                   epsilon_decay=MADQN_config.EPSILON_DECAY,
                   optimizer_type="rmsprop", training_strategy=MADQN_config.TRAINING_STRATEGY,
                   model_type=MADQN_config.MODEL_TYPE, outputs_dir=test_dir)
    else:
        raise ValueError("no valid rl option")

    rl.load(directory=os.path.join(outputs_dir, "models"), check_point=check_point)

    write_to_log(f"RL option: {rl_option}\n"
                 f"Agents: {env.n_agents}\n"
                 f"Actions:{env.n_action}\n"
                 f"Training episodes:{check_point * MAA2C_config.EVAL_INTERVAL}", output_dir=test_dir)

    evaluation(env, rl, output_dir=test_dir,eval_episodes=eval_episodes)
    write_to_log("END------------------------------------------", output_dir=test_dir)


def evaluation(env, rl, eval_episodes=100, output_dir="/logs", render=False):

    avg_total_reward = []
    avg_speed = []
    avg_trip_delays = []
    avg_headways = []
    avg_ttcs = []
    collisions_rate = []

    for i in range(eval_episodes):

        rewards = [[]] * env.n_agents
        speeds = []
        ttcs = []
        headways = []
        trip_time = []
        global_reward = []
        local_reward = []

        states, _ = env.reset(show_gui=True)
        eval_done = False
        while not eval_done:
            action = rl.act(env.normalize_state(states))
            new_states, step_rewards, eval_done, info = env.step(action)
            env.render(episode=i, output_dir=output_dir) if render else None
            global_reward.append(info["global_rewards"])
            local_reward.append(info["local_rewards"])

            # env.render(eval_number)
            for agent in range(0, env.n_agents):
                rewards[agent].append(step_rewards[agent])
                speeds.append(states[agent][2]) if states[agent][2] > 0 else None
                ttcs.append(states[agent][6])  # if states[agent][6] < INFINITY else None
                headways.append(states[agent][7])  # if states[agent][7] < INFINITY else None
                trip_time.append(states[agent][8])  # if states[agent][7] < INFINITY else None
            states = new_states
        rl.tensorboard.step = i
        env.close()

        rewards_mu, rewards_std, _, _ = agg_stat_list(rewards)
        glob_sum = np.sum(np.array(global_reward).flatten())
        locl_avg = np.sum(np.mean(np.array(local_reward), axis=1))

        rl.tensorboard.update_stats(
            {
                "reward_avg": rewards_mu,
                "global_reward_sum": glob_sum,
                "local_reward_avg": locl_avg,
                "reward_std": rewards_std,
                "speed_avg": np.mean(speeds),
                "min_ttc": np.min(ttcs),
                "headway": np.min(headways),
                "trip_time": np.max(trip_time),
                "collisions": len(env.collided_vehicles)
            }
        )
        ttcs = np.array(ttcs)
        ttcs = ttcs[ttcs < 1000]
        avg_total_reward.append(rewards_mu)
        avg_speed.append(np.mean(speeds))
        avg_ttcs.append(np.min(ttcs)) if len(ttcs) != 0 else None
        avg_headways.append(np.min(headways))
        avg_trip_delays.append(np.max(trip_time))
        collisions_rate.append(len(env.collided_vehicles))

    write_to_log(f"avg total reward {np.mean(avg_total_reward)}\n"
                 f"avg speed {np.mean(avg_speed)}\n"
                 f"avg min ttc {np.mean(avg_ttcs)}\n"
                 f"avg min headway {np.mean(avg_headways)}\n"
                 f"avg min trip delay {np.mean(avg_trip_delays)}\n"
                 f"collisions rate {np.sum(collisions_rate)/ eval_episodes}\n", output_dir=output_dir)


def evaluate_policy_behavior(env, rl):
    # rewards = [[]] * env.n_agents
    # speeds = [[]] * env.n_agents
    # ttcs = [[]] * env.n_agents
    # headways = [[]] * env.n_agents
    # trip_delays = [[]] * env.n_agents
    states, _ = env.reset(show_gui=True)
    eval_done = False
    step = 0
    while not eval_done:
        action = rl.act(env.normalize_state(states))
        new_states, step_rewards, eval_done, info = env.step(action)
        # env.render(eval_number)
        for agent in range(0, env.n_agents):
            reward = step_rewards[agent]
            speed = states[agent][2]
            edge_id = states[agent][3]
            ttc = states[agent][6]  # if states[agent][6] < INFINITY else None
            headway = states[agent][7]  # if states[agent][7] < INFINITY else None
            trip_delay = states[agent][8]  # if states[agent][7] < INFINITY else None

            rl.tensorboard.update_stats(
                {
                    f"reward {agent}": reward,
                    f"edge_id {agent}": edge_id,
                    f"speed {agent}": speed,
                    f"ttc {agent}": ttc,
                    f"headway {agent}": headway,
                    f"trip delay {agent}": trip_delay,
                }
            )

            # rewards[agent].append(reward)
            # speeds[agent].append(speed) if speed > 0 else None
            # ttcs[agent].append(ttc)
            # headways[agent].append(headway)
            # trip_delays[agent].append(trip_delay)

        states = new_states
        step += 1
        rl.tensorboard.step = step

    env.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
