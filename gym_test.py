import gym
import numpy as np
from matplotlib import pyplot as plt

from MARL.agent.A2C import A2C


def main():
    env = gym.make('CartPole-v1', render_mode="human")

    # state = (pos, v, angle, Angular Velocity)
    # actions = (push left, push right)
    rl = A2C(4, 2,
             memory_capacity=100,
             reward_gamma=0.99, reward_scale=1.,
             actor_hidden_size=32, critic_hidden_size=32,
             critic_loss="mse", actor_lr=0.001, critic_lr=0.001,
             optimizer_type="rmsprop", entropy_reg=0.01,
             max_grad_norm=0.5, batch_size=100,
             epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=0.98)

    scores = []
    episodes = []
    for i in range(3000):
        observation, _ = env.reset()
        done = False
        while not done:
            action = rl.exploration_action(observation)
            new_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rl.remember(observation, action, reward, new_observation, done)
            observation = new_observation

        if i > 3:
            rl.learn()

        # evaluation
        if i % 20 == 0:
            observation, _ = env.reset()
            eva_done = False
            score = 0
            while not eva_done:
                action = rl.action(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                score += reward
                env.render()
                eva_done = terminated or truncated
            scores.append(score)
            episodes.append(i)

    print(f"end training... mean scores = {np.mean(scores)}")
    env.close()

    episodes = np.array(episodes)
    eval_rewards = np.array(scores)

    plt.figure()
    plt.plot(episodes, eval_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend(["A2C"])
    plt.show()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
