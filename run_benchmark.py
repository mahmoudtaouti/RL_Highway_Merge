from MARL.MAA2C import MAA2C
from MARL.common.utils import agg_double_list
from on_ramp_env import OnRampEnv
import argparse

MAX_EPISODES = 50
EPISODES_BEFORE_TRAIN = 8
EVAL_EPISODES = 1
EVAL_INTERVAL = 5

MEMORY_CAPACITY = 1000
BATCH_SIZE = 32
CRITIC_LOSS = "mse"
MAX_GRAD_NORM = 0.5

REWARD_DISCOUNTED_GAMMA = 0.99
EPSILON_START = 0.99
EPSILON_END = 0.01
EPSILON_DECAY = 0.01


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
               memory_capacity=MEMORY_CAPACITY, batch_size=BATCH_SIZE,
               reward_gamma=REWARD_DISCOUNTED_GAMMA,
               actor_hidden_size=256, critic_hidden_size=256,
               epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
               epsilon_decay=EPSILON_DECAY,
               optimizer_type="adam")

    for eps in range(MAX_EPISODES):
        state, _ = env.reset(opt.show_gui, opt.sync_with_carla)
        done = False
        step = 0
        rl.tensorboard.step = eps
        while not done:
            step += 1
            # select agents action
            actions = rl.exploration_act(state, n_episodes=eps)

            # perform actions on env
            new_state, reward, done, _ = env.step(actions)

            # global reward for each agent
            reward = [reward] * len(env.controlled_vehicles)

            # remember experience
            rl.remember(state, actions, reward, new_state, done)

        if eps > EPISODES_BEFORE_TRAIN:
            rl.learn()

        env.close()

        if eps % EVAL_INTERVAL == 0:

            rewards = []
            speeds = []
            ttcs = []
            headways = []
            infos = []

            for i in range(EVAL_EPISODES):
                rewards_i = []
                infos_i = []
                state, _ = env.reset(show_gui=True)
                eval_done = False
                while not eval_done:
                    action = rl.act(state)
                    state, reward, eval_done, info = env.step(action)
                    # if(step % 5 == 0):
                    #    self.env.render(eval_num) if i == 0 else None
                    rewards_i.append(reward)
                    infos_i.append(info)
                    for agent in range(0, env.n_agents):
                        speeds.append(state[agent][2])
                        ttcs.append(state[agent][7])
                        headways.append(state[agent][8])
                env.close()
                rewards.append(rewards_i)
                infos.append(infos_i)

            rewards_mu, rewards_std, r_max, r_min = agg_double_list(rewards)

            rl.tensorboard.update_stats(
                reward_avg=rewards_mu,
                reward_std=rewards_std,
                epsilon=rl.epsilon)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
