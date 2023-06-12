import argparse
import os

import numpy as np

import config as cnf
from MARL.MAA2C import MAA2C
from MARL.MADQN import MADQN
from MARL.common.utils import agg_list_stat
from on_ramp_env import OnRampEnv
from util.ModifiedTensorBoard import ModifiedTensorBoard
from util.common_util import write_to_log, to_ndarray


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

    check_point = 97
    outputs_dir = "./outputs/Normal_Running"
    test_dir = os.path.join(outputs_dir, "benchmark")
    os.makedirs(test_dir, exist_ok=True)

    env = OnRampEnv(exec_num=check_point)

    #
    # rl = MAA2C(n_agents=env.n_agents, state_dim=env.n_state, action_dim=env.n_action,
    #            memory_capacity=cnf.MEMORY_SIZE, batch_size=cnf.BATCH_SIZE,
    #            reward_gamma=cnf.REWARD_DISCOUNTED_GAMMA,
    #            actor_hidden_size=cnf.ACTOR_HIDDEN_SIZE, critic_hidden_size=cnf.CRITIC_HIDDEN_SIZE,
    #            epsilon_start=cnf.EPSILON_START, epsilon_end=cnf.EPSILON_END,
    #            epsilon_decay=cnf.EPSILON_DECAY,
    #            optimizer_type=cnf.OPTIMIZER_TYPE, training_strategy=cnf.TRAINING_STRATEGY, is_evaluation=True)

    # rl = MADQN(n_agents=env.n_agents, state_dim=env.n_state, action_dim=env.n_action,
    #            memory_capacity=cnf.MEMORY_SIZE, batch_size=cnf.BATCH_SIZE,
    #            target_update_freq=50, reward_gamma=cnf.REWARD_DISCOUNTED_GAMMA,
    #            actor_hidden_size=cnf.ACTOR_HIDDEN_SIZE, critic_loss=cnf.CRITIC_LOSS,
    #            epsilon_start=cnf.EPSILON_START, epsilon_end=cnf.EPSILON_END, epsilon_decay=cnf.EPSILON_DECAY,
    #            optimizer_type="rmsprop", training_strategy=cnf.TRAINING_STRATEGY, model_type=cnf.MODEL_TYPE)

    # rl.load(directory=os.path.join(outputs_dir, "models"), check_point=check_point)
    # tensorboard = ModifiedTensorBoard(test_dir)

    state, _ = env.reset(opt.show_gui, opt.sync_with_carla)
    done = False
    rewards = [[]] * env.n_agents
    speeds = [[]] * env.n_agents
    ttcs = [[]] * env.n_agents
    headways = [[]] * env.n_agents
    trip_time_delays = [[]] * env.n_agents

    step = 0
    write_to_log(f"Normal Running\n"
                 f"Number of agents: {env.n_agents}\n"
                 f"Number of actions:{env.n_action}\n"
                 f"Number of episodes:{check_point * cnf.EVAL_INTERVAL}", output_dir=test_dir)

    while not done:
        step += 1
        # actions = rl.act(state)
        # perform actions on env
        new_state, glob, done, info = env.step([0, 0, 0, 0, 0, 0])
        # env.render()
        rewards_i = [locl_r + glob for locl_r in info["local_rewards"]]
        # env.render(output_dir=test_dir, episode=2)
        for agent in range(env.n_agents):
            rewards[agent].append(rewards_i[agent])
            speeds[agent].append(state[agent][2])
            ttcs[agent].append(state[agent][6])
            headways[agent].append(state[agent][7])
            trip_time_delays[agent].append(state[agent][8])
        # tensorboard.update_stats(speed_veh0=speeds[0][-1], speed_vh1=speeds[1][-1])
        state = new_state
        # tensorboard.step = step

    mean_rewards, _, _, _ = agg_list_stat(rewards)
    ttcs = to_ndarray(ttcs)
    ttcs = [ttc[ttc > 0] for ttc in ttcs]
    headways = to_ndarray(headways)
    headways = [headway[headway > 0] for headway in headways]

    write_to_log(f"Reward average: {mean_rewards}\n"
                 f"Speed average: {np.mean(speeds)}\n"
                 f"Maximum trip time delay: {np.max(trip_time_delays)}\n"
                 f"Minimum headway: {np.min(headways)}\n"
                 f"Minimum TTC: {np.min(ttcs)}\n"
                 f"Collided vehicles: {len(env.collided_vehicles)}", output_dir=test_dir)
    write_to_log(f"%%%", output_dir=test_dir)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
