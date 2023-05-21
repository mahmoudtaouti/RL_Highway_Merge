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
import util.commun_util as c_util


frame_rate_steps = 3

def arg_parse():
    # create an argument parser
    parser = argparse.ArgumentParser(description="Zipper merge e1 runner.")
    
    #parser.add_argument('-m', "--tensor_model", required=True, type=str,
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

# evaluation the learned agents
def evaluation(env, rl, eval_episodes=10,eval_num=0):
    rewards = []
    infos = []
    for i in range(eval_episodes):
        rewards_i = []
        infos_i = []
        state, _ = env.reset(show_gui = True)
        action = action(state)
        state, reward, done, info = env.step(action)
        done = done[0] if isinstance(done, list) else done
        rewards_i.append(reward)
        infos_i.append(info)
        step =0
        while not done:
            step+=1
            action = rl.action(state)
            state, reward, done, info = env.step(action)
            #if(step % 5 == 0):
            #    self.env.render(eval_num) if i == 0 else None
            done = done[0] if isinstance(done, list) else done
            rewards_i.append(reward)
            infos_i.append(info)
        rewards.append(rewards_i)
        infos.append(infos_i)
    #env.close()
    return rewards, infos

def main():
    # parse the arguments
    opt = arg_parse()
    # create environment and agent
    env = OnRampEnv()
    
    
    state = env.reset(opt.show_gui, opt.sync_with_carla) 
    done = False
    total_reward = []
    step = 0
    
    # Create a folder to store the screenshots
    output_folder = "./outputs/100000"
    os.makedirs(output_folder, exist_ok=True)
    
    while not done:
        step +=1
        # select agents action
        random_index = np.random.randint(len(env.action_space))
        
        a_1 = 2 if step > 170 and step < 180 else 1
        a_1 = a_1 if step < 180 else 0
        a_1 = 4 if step == 120 else a_1
        a_2 = 0 if step > 170 else 1
        
        actions = (1,1)
        
        # perform actions on env
        new_state, reward, done, _ = env.step(actions)
        
        total_reward.append(reward)
        state = new_state
    
    #if env.show_gui:
    #    c_util.convert_images_to_video(output_folder, f"{output_folder}/VID1000.mp4")
    
    print(f"Vehicle: {env.controlled_vehicles} - AVG_reward: {sum(total_reward)} - finished with {env.finished_at} steps")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')