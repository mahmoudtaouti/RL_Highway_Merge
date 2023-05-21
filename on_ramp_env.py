import glob
import os
import sys

import numpy as np
import math
import itertools

import traci
import sumolib
import config as cnf

import random
import time
from sync_simulation import TraCiSync
import util.traci_util as tr_util
import util.commun_util as c_util

# state(posX, posY, speed(m/s), edge_id, lane_indx, dist_merge_node, merge_node_visibility(0 or 1), TTC,hedway,tip_time)
# state space indexes to access val
s_posX = 0
s_posY = 1
s_speed = 2
s_edge = 3
s_lane = 4
s_dist_merg_nod = 5
s_merg_visib = 6
s_ttc = 7
s_headway = 8
s_trip_time = 9

# action ['idle', 'accelerate', 'decelerate','change_right', 'change_left']
# action indexes to access val
a_idle = 0
a_acc = 1
a_dec = 2
a_right = 3
a_left = 4


class OnRampEnv:
    show_gui = False
    syn_with_carla = False
    step_num = 0
    min_steps = 80  # all vehicles should be departed, min_step =  (depart/DELTA_SEC) + 1
    max_steps = 550

    # simulation var
    on_ramp_edge = "44.0.00"
    terminal_edge = "38.0.00"
    merging_node = "720"
    highway_edges = ("40.0.00", "39.0.00", "38.0.00")
    observation_range = 150  # assume AVs can detect objects within a range of 150 meters

    # agents var
    controlled_vehicles = ["0", "1"]
    TTC_threshold = 25

    #
    on_ramp_properties = {
        'd_speed': 9,
        'max_speed': 15,
        'speed_range': (0, 15),
        'min_gap': 10,
        'hard_braking': -7,
    }

    #
    on_highway_properties = {
        'd_speed': 25,
        'max_speed': 30,
        'speed_range': (0, 30),
        'min_gap': 10,
        'min_speed': 15,
        'hard_braking': -7
    }

    s_edge_mapping = {
        '40.0.00': 0,
        '44.0.00': 1,
        '39.0.00': 2,
        '38.0.00': 3,
        ':720_0': 4,
        ':720_1': 5,
        ':720_2': 6,
        ':720_3': 7,
        ':720_4': 8,
        ':720_5': 9,
        ':720_6': 10,
        ':669_0': 11,
        ':669_1': 12,
        ':669_2': 13,
        ':669_3': 14,
        ':669_4': 16,
        ':669_5': 16,
        ':669_6': 16
    }

    s_merg_visib_mapping = {
        False: 0,
        True: 1
    }

    def __init__(self, exec_num=1e3):

        self.agent_actions = ['idle', 'accelerate', 'decelerate', 'change_right', 'change_left']
        # Define the joint action space
        joint_action_space = list(itertools.product(self.agent_actions, repeat=2))
        self.action_space = np.array(joint_action_space)
        self.init_state_vals = (0, 0, 0, 0, 0, 9999, 0, 9999, 9999, 9999)
        self.state_space = [self.init_state_vals for _ in self.controlled_vehicles]
        self.traci_sync = None
        self.exec_num = exec_num
        self.n_state = len(self.init_state_vals)
        self.n_action = len(self.agent_actions)

    def reset(self, show_gui=False, syn_with_carla=False):

        self.show_gui = show_gui
        self.syn_with_carla = syn_with_carla
        self.step_num = 0

        try:
            self.close()
        except:
            pass

        self._starSIM()

        # depart steps (all vehicles entered the sim)
        while self.step_num < self.min_steps:
            self.sim_step()
            self.step_num += 1

        # disable automatic control by SUMO.
        for vehID in self.controlled_vehicles:
            if self.agent_is_exist(vehID):
                traci.vehicle.setSpeedMode(vehID, 32)
                traci.vehicle.setLaneChangeMode(vehID, 512)
                traci.vehicle.setAccel(vehID, 0)
            else:
                raise AssertionError(f"Make sure vehicle {vehID} is departed")

        self._update_state()

        state = c_util.to_ndarray(self.state_space)
        return state, {}

    def step(self, actions):

        done = False
        reward = 0
        info = {}

        info["agents_dones"] = tuple(
            not self.agent_is_exist(veh) or self.agent_is_arrived(veh) for veh in self.controlled_vehicles)

        # end episode
        if all(info["agents_dones"]) or self.step_num >= self.max_steps or traci.simulation.getMinExpectedNumber() == 0:
            done = True
            self.step_num = 0
            self.finished_at = self.step_num
            return c_util.to_ndarray(self.state_space), reward, done, info

        # before take actions
        state = self.state_space
        info["agents_actions"] = actions
        # take actions and advance time
        callback = lambda: self._action(state, actions)
        self.sim_step(callback)

        # Get new state and reward
        new_state = self._update_state()
        reward = self._reward(state, actions, new_state)

        # local reward
        # info["regional_rewards"] = tuple(self._local_rewards(state, actions, new_state))
        new_state = c_util.to_ndarray(new_state)
        info["agents_info"] = new_state
        self.step_num += 1
        return new_state, reward, done, info

    def _reward(self, state, actions, new_state):
        # Cooperative multi-agent reward
        # Combine local and global rewards using appropriate weighting or aggregation scheme
        # adjust the weights based on the importance of local vs. global rewards
        glob = self._global_reward(state, actions, new_state)
        locl = sum(self._local_rewards(state, actions, new_state))
        return 0.7 * glob + 0.5 * locl

    def _global_reward(self, state, actions, new_state):
        # Calculate global rewards based on the state of the entire system
        # Consider system-level objectives such as trip time delay, safety, efficiency, etc.
        reference_trip_time = 60  # reference to maximum trip time for all vehicles could take

        reward = 0
        cost = 0

        for indx, veh in enumerate(self.controlled_vehicles):
            curr_trip_time = state[indx][s_trip_time]
            if self.agent_is_collide(veh):
                cost += 200
            if self.agent_is_exist(veh) and self.agent_is_arrived(veh):
                reward += -1 * min(0, curr_trip_time - reference_trip_time)

        return reward - cost

    def _local_rewards(self, state, actions, new_state):
        rewards = []
        for indx, (vehicle, action) in enumerate(zip(self.controlled_vehicles, actions)):

            if not self.agent_is_exist(vehicle):
                rewards.append(0)
            else:
                if state[indx][s_edge] == self.s_edge_mapping[self.on_ramp_edge]:
                    ''' vehilces on ramp '''
                    rewards.append(self._on_ramp_reward(indx, vehicle, state, action, new_state))
                else:
                    ''' vehicles on highway'''
                    rewards.append(self._on_highway_reward(indx, vehicle, state, action, new_state))
        return rewards

    def _on_ramp_reward(self, indx, agent, state, action, new_state):

        # init
        reward = 0
        cost = 0
        current_speed = state[indx][s_speed]
        new_speed = new_state[indx][s_speed]
        d_speed = self.on_ramp_properties["d_speed"]
        speed_range = self.on_ramp_properties["speed_range"]
        min_gap = self.on_ramp_properties['min_gap']
        scaled_speed = c_util.lmap(state[indx][s_speed], speed_range, [0, 1])
        headway = new_state[indx][s_headway]
        ramp_edge = self.s_edge_mapping[self.on_ramp_edge]

        # cost for illegal actions
        cost += 5 if not self._action_is_legal(agent, action) else 0

        # height speed reward
        reward += np.clip(scaled_speed, 0, 1)

        # maintaining good stable speed
        cost += abs(new_speed - current_speed)

        if action == a_dec:
            # Reward for preserving minimum gap
            if headway >= min_gap:
                reward = 0.2
            else:
                # cost for low speed
                cost = 1.5 * (d_speed - new_speed) / d_speed if new_speed < d_speed else 0

        # Reward for safe merging with d_speed (no abrupt maneuvers)
        reward += 5 \
            if new_state[indx][s_edge] != ramp_edge else 0
        return reward - cost

    def _on_highway_reward(self, indx, agent, state, action, new_state):

        # init
        reward = 0
        cost = 0
        current_speed = state[indx][s_speed]
        new_speed = new_state[indx][s_speed]
        d_speed = self.on_highway_properties["d_speed"]
        speed_range = self.on_highway_properties["speed_range"]
        scaled_speed = c_util.lmap(new_speed, speed_range, [0, 1])
        # TODO action reward only on visible observation range
        ttc = tr_util.TTC_to_ramp_veh(agent, self.on_ramp_edge)  # \
        #    if new_state[indx][s_merg_visib] == 1 else 9999

        # cost for illegal actions
        cost += 5 if not self._action_is_legal(agent, action) else 0

        # reward for height speed
        reward += np.clip(scaled_speed, 0, 1)

        # maintaining stable speed
        cost += abs(new_speed - current_speed)

        if action == a_idle:
            pass
        elif action == a_acc:
            pass
        elif action == a_dec:
            if ttc < self.TTC_threshold:
                reward += 10
            # decelerate on low speed cost (no reason)
            cost += 2 * (d_speed - new_speed) / d_speed if new_speed < d_speed else 0

        else:
            # TODO check right lae change
            # change lane to left (no abrupt maneuvers)
            cost += 0 if tr_util.change_lane_chance(agent, change_diraction=1) else -0.5
            if action == a_left:
                # reward for change lane to left to allow merge
                reward += 2 if ttc < self.TTC_threshold else 0
        return reward - cost

    def _update_state(self):
        # TODO round float values they are  too long
        """
        state(posX, posY, speed(m/s), edge_id, lane_indx, dist_merge_node, merge_node_visibility(0 or 1), TTC,
        headway, trip_time)
        """

        state = []
        for veh in self.controlled_vehicles:
            if not self.agent_is_exist(veh):
                state.append(self.init_state_vals)
            elif self.agent_is_collide(veh):
                state.append(self.init_state_vals)
                traci.vehicle.remove(veh)
            else:
                posX = traci.vehicle.getPosition(veh)[0]
                posY = traci.vehicle.getPosition(veh)[1]
                sp = traci.vehicle.getSpeed(veh)
                rID = self.s_edge_mapping[traci.vehicle.getRoadID(veh)]
                lane = traci.vehicle.getLaneIndex(veh)
                dis_m = tr_util.get_distance_to_merge_point(veh, self.merging_node)
                mer_pnt_visib = self.s_merg_visib_mapping[dis_m < self.observation_range]
                ttc = tr_util.TTC_to_ramp_veh(veh, self.on_ramp_edge)
                headway = tr_util.headway_distance(veh)
                trip_time = tr_util.trip_time(veh, arrival_edge=self.terminal_edge)
                state_i = [posX, posY, sp, rID, lane, dis_m, mer_pnt_visib, ttc, headway, trip_time]
                state.append(state_i)

        self.state_space = state
        return self.state_space

    def _action(self, state, actions):
        for indx, (action, vehicle) in enumerate(zip(actions, self.controlled_vehicles)):
            if self.agent_is_exist(vehicle) and self._action_is_legal(vehicle, action):
                if self.state_space[indx][s_edge] == self.s_edge_mapping[self.on_ramp_edge]:
                    '''actions for on-ramp vehicles'''
                    self._act(vehicle, action)
                else:
                    '''actions for highway vehicles'''
                    # when vehicle observe the ramp on this highway
                    dis_to_mer = tr_util.get_distance_to_merge_point(vehicle, self.merging_node)
                    self._act(vehicle, action)
                    if dis_to_mer <= self.observation_range:
                        # self.state_space[indx][s_ttc]
                        self._act(vehicle, action)
                    else:
                        self._act(vehicle, action)
            else:
                # TODO illegal action cost
                pass

    def _act(self, vehID, action):
        if action == a_idle:
            pass
        elif action == a_acc:
            tr_util.accelerate(vehID)
        elif action == a_dec:
            tr_util.decelerate(vehID)
        elif action == a_right:
            tr_util.change_to_right(vehID)
        elif action == a_left:
            tr_util.change_to_left(vehID)

    def _starSIM(self):

        seed = random.randint(1, 2000)
        if self.show_gui:
            sumoCmd = [sumolib.checkBinary('sumo-gui'),
                       "-c", "./map/Town04.sumocfg",
                       '--start',
                       '--quit-on-end',
                       '--step-length', str(cnf.DELTA_SEC),
                       '--lateral-resolution', '0.25',
                       '--seed', str(seed)]
        else:
            sumoCmd = [sumolib.checkBinary('sumo'),
                       "-c", "./map/Town04.sumocfg",
                       '--step-length', str(cnf.DELTA_SEC),
                       '--lateral-resolution', '0.25',
                       '--seed', str(seed)]

        if self.syn_with_carla:
            self.traci_sync = TraCiSync()
        else:
            traci.start(sumoCmd)

        if self.show_gui:
            viewID = traci.gui.DEFAULT_VIEW
            traci.gui.setZoom(viewID, zoom=493.96)
            traci.gui.setOffset(viewID, x=354.106, y=361.329)

        # disable sumo collision 
        # safety, so we can make the agent by his own learn to avoid collisions
        traci.simulation.setParameter("", "collision.actionLaneChange", "none")
        traci.simulation.setParameter("", "collision.check-junctions", "none")

    def _action_is_legal(self, vehicle, action):

        # TODO use the state instead of getting values from traci
        if not self.agent_is_exist(vehicle):
            return False

        road_id = traci.vehicle.getRoadID(vehicle)
        lane_number = traci.vehicle.getLaneIndex(vehicle)
        if action == a_left:
            if road_id == self.on_ramp_edge:
                return False
            elif lane_number > 1:
                return False

        elif action == a_right:
            if road_id == self.on_ramp_edge:
                return False
            elif lane_number < 2:
                return False

        # hard braking for highway and ramp
        if road_id == self.on_ramp_edge:
            if traci.vehicle.getAcceleration(vehicle) < self.on_ramp_properties['hard_braking']:
                return False
        else:
            if traci.vehicle.getAcceleration(vehicle) < self.on_highway_properties['hard_braking']:
                return False

        return True
        # TODO implement speed related illegal actions

    def agent_is_exist(self, vehicle):
        return vehicle in traci.vehicle.getIDList()

    def agent_is_arrived(self, vehicle):
        return traci.vehicle.getRoadID(vehicle) == self.terminal_edge

    def agent_is_collide(self, vehicle):
        return vehicle in traci.simulation.getCollidingVehiclesIDList()

    def render(self, episode=0):
        output_folder = f"./outputs/{self.exec_num}/record"
        os.makedirs(output_folder, exist_ok=True)
        output_folder = f"{output_folder}/eva{episode}"
        os.makedirs(output_folder, exist_ok=True)
        try:
            traci.gui.screenshot(traci.gui.DEFAULT_VIEW, filename=f"{output_folder}/{self.step_num}.png")
        except:
            pass

    def sim_step(self, callback=None):

        if self.syn_with_carla:
            self.traci_sync.simulationStep(callback)
        else:
            if callback is not None:
                callback()
            traci.simulationStep()

    def close(self):
        if self.syn_with_carla:
            self.traci_sync.close()
        else:
            traci.close()
