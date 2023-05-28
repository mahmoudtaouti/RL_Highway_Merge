import argparse

import numpy as np

import config as cnf
from MARL.MAA2C import MAA2C
from on_ramp_env import OnRampEnv


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

    rl = MAA2C(n_agents=env.n_agents, state_dim=env.n_state, action_dim=env.n_action,
               memory_capacity=cnf.MEMORY_SIZE, batch_size=cnf.BATCH_SIZE,
               reward_gamma=cnf.REWARD_DISCOUNTED_GAMMA,
               actor_hidden_size=256, critic_hidden_size=256,
               epsilon_start=cnf.EPSILON_START, epsilon_end=cnf.EPSILON_END,
               epsilon_decay=cnf.EPSILON_DECAY,
               optimizer_type="rmsprop", training_strategy=cnf.TRAINING_STRATEGY)

    # rl.load(directory="./outputs/17/models", check_point=7)

    state, _ = env.reset(opt.show_gui, opt.sync_with_carla)
    done = False
    total_reward = []
    step = 0

    while not done:
        step += 1
        # select agents action
        random_index = np.random.randint(len(env.action_space))

        a_1 = 1 if step < 120 else 0
        a_1 = 2 if 170 > step > 140 else a_1
        a_2 = 1 if step < 90 else 0

        actions = (a_1, a_2)
        # actions = rl.act(state)

        # perform actions on env
        new_state, glob, done, info = env.step(actions)
        # env.render()
        rewards = [locl_r + glob for locl_r in info["local_rewards"]]
        print(f"local rewards {info['local_rewards']}, global {glob}")
        total_reward.append(sum(rewards))
        state = new_state

    # if env.show_gui:
    #    c_util.convert_images_to_video(output_folder, f"{output_folder}/VID1000.mp4")

    print(
        f"Vehicle: {env.controlled_vehicles} - SUM_reward: {sum(total_reward)} - finished with {env.finished_at} steps")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
