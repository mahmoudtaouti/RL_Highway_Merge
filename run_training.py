from MARL.MADQN import MADQN
from MARL.MAA2C import MAA2C
from MARL.common.utils import agg_double_list
from util.commun_util import increment_counter
import os
from on_ramp_env import OnRampEnv
import util.commun_util as c_util
import argparse
import numpy as np
import matplotlib.pyplot as plt

MAX_EPISODES = 1500
EPISODES_BEFORE_TRAIN = 10
EVAL_EPISODES = 1
EVAL_INTERVAL = 3

MAX_STEPS = 10000

MEMORY_CAPACITY = 1000
BATCH_SIZE = 128
CRITIC_LOSS = "mse"
MAX_GRAD_NORM = None

REWARD_DISCOUNTED_GAMMA = 0.99
EPSILON_START = 0.99
EPSILON_END = 0.01
EPSILON_DECAY = 0.003


def main():
    # parse the arguments
    # opt = arg_parse()

    rl_option = "MADQN"
    training_strategy = "centralized"
    exec_num = increment_counter()

    env = OnRampEnv(exec_num=exec_num)
    state_dim = env.n_state
    action_dim = env.n_action

    outputs_dir = f"./outputs/{exec_num}/"
    os.makedirs(outputs_dir, exist_ok=True)

    if rl_option == "MADQN":
        rl = MADQN(env, memory_capacity=MEMORY_CAPACITY,
                   state_dim=state_dim, action_dim=action_dim,
                   batch_size=BATCH_SIZE, max_steps=MAX_STEPS,
                   reward_gamma=REWARD_DISCOUNTED_GAMMA,
                   actor_hidden_size=256, critic_hidden_size=256,
                   epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
                   epsilon_decay=EPSILON_DECAY, max_grad_norm=MAX_GRAD_NORM,
                   episodes_before_train=EPISODES_BEFORE_TRAIN, use_cuda=False,
                   reward_type="global_R", target_update_freq=20,
                   optimizer_type="adam", outputs_dir=outputs_dir)
    elif rl_option == "MAA2C":
        rl = MAA2C(env, n_agents=len(env.controlled_vehicles),
                   state_dim=state_dim, action_dim=action_dim,
                   training_strategy=training_strategy,
                   roll_out_n_steps=10, memory_capacity=MEMORY_CAPACITY,
                   batch_size=BATCH_SIZE, max_steps=MAX_STEPS,
                   reward_gamma=REWARD_DISCOUNTED_GAMMA,
                   actor_hidden_size=256, critic_hidden_size=256,
                   epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
                   epsilon_decay=EPSILON_DECAY, max_grad_norm=MAX_GRAD_NORM,
                   episodes_before_train=EPISODES_BEFORE_TRAIN, use_cuda=False,
                   optimizer_type="adam")
    else:
        raise ValueError(f"no rl option for {rl_option}")

    # env.reset()
    c_util.write_to_log(f"EXEC==================================================\n" \
                        f"Execution number : {exec_num}\n" \
                        f"RL option: {rl_option} \n" \
                        f"Training strategy: {training_strategy} \n"
                        "=======================================================", output_dir=outputs_dir)

    episodes = []
    eval_rewards = []
    infos = []
    eval_num = 0

    while rl.n_episodes < MAX_EPISODES:

        rl.interact()

        if rl.n_episodes >= EPISODES_BEFORE_TRAIN:
            rl.train()

        if rl.episode_done and ((rl.n_episodes) % EVAL_INTERVAL == 0):
            eval_num += 1
            rewards, info = rl.evaluation(EVAL_EPISODES, eval_num)

            rewards_mu, rewards_std, r_max, r_min = agg_double_list(rewards)

            rl.save(model_dir=outputs_dir, global_step=eval_num)

            print("Episode %d, Average Reward %.2f" % (rl.n_episodes + 1, rewards_mu))
            episodes.append(rl.n_episodes + 1)
            eval_rewards.append(rewards_mu)
            infos.append(info)

            # write information to log file
            c_util.write_to_log(f"EVA***********************************************************************\n" \
                                f"episodes : {rl.n_episodes + 1}\n" \
                                f"rewards : {rewards_mu}\n" \
                                f"infos :- \n", output_dir=outputs_dir)
            for i in info:  # each Eval
                for data in i:  # each step
                    try:  # not always information provided
                        c_util.write_to_log(f"\n" \
                                            f"\t\tagents dones : {data['agents_dones']}\n" \
                                            f"\t\tagents actions : {data['agents_actions']}\n" \
                                            f"\t\tagents states : {data['agents_info']}\n\n",
                                            output_dir=outputs_dir)
                    except:
                        pass

    episodes = np.array(episodes)
    eval_rewards = np.array(eval_rewards)

    plt.figure()
    plt.plot(episodes, eval_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend([rl_option])
    plt.show()


def arg_parse():
    # create an argument parser
    parser = argparse.ArgumentParser(description="On-Ramp Highway Merging Scenario runner.")

    parser.add_argument('-a', "--algorithm", required=True, type=str, choices=['MADQN', 'MAA2C'],
                        help='Choose which algorithm to train with [MADQN, MAA2C]')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
