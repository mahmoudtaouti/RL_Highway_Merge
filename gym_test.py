import gym
import numpy as np
from matplotlib import pyplot as plt
import MADQN_config as cnf
from MARL.agent.A2C import A2C
from MARL.agent.DQN import DQN
from MARL.common.utils import exponential_epsilon_decay


def main():
    env = gym.make('CartPole-v1')
    eva_env = gym.make('CartPole-v1', render_mode="human")
    # state = (pos, v, angle, Angular Velocity)
    # actions = (push left, push right)
    rl = DQN(4, 2,
             memory_capacity=500,
             reward_gamma=0.99, reward_scale=1.,
             actor_hidden_size=64,
             critic_loss="huber", actor_lr=0.001,
             optimizer_type="rmsprop",
             max_grad_norm=0.9, batch_size=200, target_update_freq=20,
             epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1)

    scores = []
    episodes = []
    epsilon = 1
    for i in range(1200):
        observation, _ = env.reset()
        done = False
        while not done:
            epsilon = exponential_epsilon_decay(epsilon_start=1, epsilon_end=0.01, decay_rate=0.0009, episode=i)
            action = rl.exploration_action(observation, epsilon=epsilon)
            new_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rl.remember(observation, action, reward, new_observation, done)
            observation = new_observation

        if i > 3:
            rl.learn()

        # evaluation
        if i % 20 == 0:
            observation, _ = eva_env.reset()
            eva_done = False
            score = 0
            while not eva_done:
                action = rl.action(observation)
                observation, reward, terminated, truncated, info = eva_env.step(action)
                score += reward
                eva_env.render()
                eva_done = terminated or truncated
            scores.append(score)
            episodes.append(i)

    print(f"end training... mean scores = {np.mean(scores)}")
    env.close()
    eva_env.close()
    episodes = np.array(episodes)
    eval_rewards = np.array(scores)

    plt.figure()
    plt.plot(episodes, eval_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend(["DQN_bh"])
    plt.show()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
