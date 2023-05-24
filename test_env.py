import config as cnf
import time
from statistics import mean
from tqdm import tqdm
from on_ramp_env import OnRampEnv
import os
import argparse
import numpy as np
import random
import traci
from util.sumo_rec import SumoRecorder
import util.common_util as c_util


def arg_parse():
    # create an argument parser
    parser = argparse.ArgumentParser(description="Zipper merge e1 runner.")

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
    opt = parser.parse_args()
    return opt


def main():
    # parse the arguments
    opt = arg_parse()
    # create environment and agent
    env = OnRampEnv()

    state = env.reset(True, opt.sync_with_carla)
    done = False
    total_reward = []
    step = 0

    while not done:
        step += 1
        # select agents action
        random_index = np.random.randint(len(env.action_space))

        a_1 = 2 if 170 < step < 180 else 1
        a_1 = a_1 if step < 130 else 0
        a_1 = 2 if 135 > step > 120 else a_1
        a_2 = 0 if step > 100 else 1

        actions = (a_1, a_2)

        # perform actions on env
        new_state, glob, done, info = env.step(actions)
        rewards = [locl_r + glob for locl_r in info["local_rewards"]]
        print(f"local rewards {info['local_rewards']}, global {glob}")
        total_reward.append(sum(rewards))
        state = new_state

    # if env.show_gui:
    #    c_util.convert_images_to_video(output_folder, f"{output_folder}/VID1000.mp4")

    print(
        f"Vehicle: {env.controlled_vehicles} - AVG_reward: {sum(total_reward)} - finished with {env.finished_at} steps")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')