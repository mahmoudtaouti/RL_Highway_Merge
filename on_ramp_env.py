import math
import os

import numpy as np
import itertools

import traci
import sumolib
import config as cnf

import random
from sync_simulation import TraCiSync
import util.traci_util as tr_util
import util.common_util as c_util

# state
# (posX, posY, speed(m/s), edge_id, lane_indx, dist_merge_node, merge_node_visibility(0or1),TTC,headway,tip_time_delay)
# state space indexes to access val
s_posX = 0
s_posY = 1
s_speed = 2
s_edge = 3
s_lane = 4
s_dist_mrg_nod = 5
s_mrg_visib = 6
s_ttc = 7
s_headway = 8
s_trip_t_delay = 9

# action ['idle', 'accelerate', 'decelerate','change_right', 'change_left']
# action indexes to access val
a_idle = 0
a_acc = 1
a_dec = 2
a_right = 3
a_left = 4


class OnRampEnv:
    """
    On-ramp vehicles try to merge safely into the Highway
    - using Town04.rou.xml
    - Run with Sumo gui use : reset(show_gui=True)
    - Run Sync with Sumo gui and Carla use: reset(show_gui=True, syn_with_carla=True)
    - controlled_vehicles = [vehicles_id]
    Note:
        * speed m/s
        * distance meter
        * close the environment after done with close()
        * start_step = (last_depart/DELTA_SEC) + 1, all vehicles should departed before start doing .step(actions)
    """
    show_gui = False
    syn_with_carla = False
    step_num = 0
    start_step = 34
    max_steps = 600

    # simulation var
    on_ramp_edge = "44.0.00"
    terminal_edge = "38.0.00"
    merging_node = "720"
    highway_edges = ("40.0.00", "39.0.00", "38.0.00")

    # agents var
    controlled_vehicles = ["0", "1"]
    TTC_threshold = 10
    headway_threshold = 1.2
    observation_range = 150  # assume AVs can detect objects within a range of 150 meters
    communication_range = 2000  # assume AVs using V2V communication with 5G (that's just expected reference range)

    #
    on_ramp_properties = {
        'd_speed': 18,
        'max_speed': 27,
        'min_speed': 8,
        'ttc_threshold': 10,
        'speed_range': (0, 34),
        'min_gap': 10,
        'hard_braking': -7,
    }

    #
    on_highway_properties = {
        'd_speed': 25,
        'max_speed': 41,
        'min_speed': 15,
        'speed_range': (0, 60),
        'ttc_threshold': 10,
        'min_gap': 15,
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
        ':669_6': 17
    }

    s_mrg_visib_mapping = {
        False: 0,
        True: 1
    }

    def __init__(self, exec_num=1e3):

        self.init_state_vals = (0, 0, 0, 0, 0, -1, 0, -1, -1, 0)
        self.state = [self.init_state_vals for _ in self.controlled_vehicles]

        self.agent_actions = ['idle', 'accelerate', 'decelerate', 'change_right', 'change_left']
        # Define the joint action space
        joint_action_space = list(itertools.product(self.agent_actions, repeat=2))
        self.action_space = np.array(joint_action_space)

        self.n_state = len(self.init_state_vals)
        self.n_action = len(self.agent_actions)
        self.n_agents = len(self.controlled_vehicles)

        self.traci_sync = None
        self.exec_num = exec_num
        self.finished_at = 0.0

        self.arrived_vehicles_state = {"vehicles": [], "terminal_state": []}

    def reset(self, show_gui=False, syn_with_carla=False):

        self.show_gui = show_gui
        self.syn_with_carla = syn_with_carla
        self.step_num = 0

        # try:
        #     self.close()
        # except:
        #     pass

        self._starSIM()

        # depart steps (all vehicles entered the sim)
        while self.step_num < self.start_step:
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

        state = c_util.to_ndarray(self.state)
        return state, {}

    def step(self, actions):
        """
        take one-step
        Returns:
            new_states, rewards, done, info {"agents_dones" "agents_actions" "local_rewards" "agents_info"}
        """
        done = False

        info = {
            "agents_dones": tuple(not self.agent_is_exist(veh) for veh in self.controlled_vehicles),
            "agents_actions": actions,
            "local_rewards": [0]*self.n_agents,
            "agents_info": self.state,
        }

        # end episode
        if all(info["agents_dones"]) or self.step_num >= self.max_steps or traci.simulation.getMinExpectedNumber() == 0:
            done = True
            self.finished_at = self.step_num
            self.step_num = 0
            return c_util.to_ndarray(self.state), 0, done, info

        # before take actions
        state = self.state

        # take actions and advance time
        self.sim_step(lambda: self._action(state, actions))

        # Get new state and reward
        new_state = self._update_state()
        global_reward, local_rewards = self._reward(state, actions, new_state)

        info["local_rewards"] = local_rewards
        new_state = c_util.to_ndarray(new_state)
        info["agents_info"] = new_state

        self.step_num += 1
        return new_state, global_reward, done, info

    def _reward(self, state, actions, new_state):
        # multi-agent reward
        # Combine local and global rewards using appropriate weighting or aggregation scheme
        # adjust the weights based on the importance of local vs. global rewards
        glob = self._global_reward(state, actions, new_state)
        locl = self._local_rewards(state, actions, new_state)
        locl = [l_r for l_r in locl]
        return glob, locl

    def _global_reward(self, state, actions, new_state):
        # Calculate global rewards based on the state of the entire system
        # Consider system-level objectives such as trip time delay, safety, efficiency, etc.
        reward = 0.0
        cost = 0.0
        reference_trip_time_delay = 15  # reference to maximum trip time delay for all vehicles could take

        for indx, veh in enumerate(self.controlled_vehicles):

            if self.agent_is_collide(veh):
                cost += cnf.COLLISION_COST

            for i, arr_veh in enumerate(self.arrived_vehicles_state["vehicles"]):
                if arr_veh == veh:
                    t_state = self.arrived_vehicles_state["terminal_state"][i]
                    reward += 2 if reference_trip_time_delay - t_state[s_trip_t_delay] > 0 else 0
                    print(f"trip time delay {t_state[s_trip_t_delay]}")
                    # delete vehicle from list after passing the reward
                    del self.arrived_vehicles_state["vehicles"][i]
                    del self.arrived_vehicles_state["terminal_state"][i]

        return reward - cost

    def _local_rewards(self, state, actions, new_state):
        rewards = []
        for indx, (vehicle, action) in enumerate(zip(self.controlled_vehicles, actions)):

            if not self.agent_is_exist(vehicle):
                rewards.append(0)
            else:
                if state[indx][s_edge] == self.s_edge_mapping[self.on_ramp_edge]:
                    ''' vehicles on ramp '''
                    rewards.append(self._on_ramp_reward(indx, vehicle, state, action, new_state))
                else:
                    ''' vehicles on highway'''
                    rewards.append(self._on_highway_reward(indx, vehicle, state, action, new_state))
        return rewards

    def _on_ramp_reward(self, indx, agent, state, action, new_state):

        # init
        reward = 0.0
        cost = 0.0
        current_speed = state[indx][s_speed]
        new_speed = new_state[indx][s_speed]
        d_speed = self.on_ramp_properties["d_speed"]
        speed_range = self.on_ramp_properties["speed_range"]
        scaled_speed = c_util.lmap(new_speed, speed_range, [0, 1])
        min_gap = self.on_ramp_properties['min_gap']
        max_speed = self.on_ramp_properties['max_speed']
        headway = new_state[indx][s_headway]
        ramp_edge = self.s_edge_mapping[self.on_ramp_edge]

        # cost for illegal actions
        cost += 3 if not self._action_is_legal(agent, action) else 0

        # reward for maintaining good speed
        # best case for d_seed + 4 > new_speed > d_seed - 4
        reward += min(1, 4 / (abs((new_speed - d_speed) + 1.99999)))

        # assigning delta ratio to very low or high speed
        delta_speed = (new_speed - d_speed) / d_speed
        if delta_speed < 0:
            cost += 0.4 * abs(delta_speed)
        elif new_speed > max_speed:
            cost += delta_speed

        # Reward for safe merging with d_speed (no abrupt maneuvers)
        reward += cnf.MERGING_LANE_REWARD / abs(delta_speed) if new_state[indx][s_edge] != ramp_edge else 0

        if headway != -1:
            r_headway = math.log(headway / (self.headway_threshold * current_speed))
            cost += - cnf.HEADWAY_COST * r_headway if r_headway < 0 else 0

        return reward - cost

    def _on_highway_reward(self, indx, agent, state, action, new_state):

        # init
        reward = 0.0
        cost = 0.0
        current_speed = state[indx][s_speed]
        new_speed = new_state[indx][s_speed]
        d_speed = self.on_highway_properties["d_speed"]
        speed_range = self.on_highway_properties["speed_range"]
        max_speed = self.on_highway_properties['max_speed']
        min_gap = self.on_highway_properties['min_gap']
        headway = new_state[indx][s_headway]
        scaled_speed = c_util.lmap(new_speed, speed_range, [0, 1])
        ttc = tr_util.ttc_with_ramp_veh(agent, self.on_ramp_edge)

        # reward for maintaining good speed
        # best case for d_seed + 4 > new_speed > d_seed - 4
        reward += min(1, 4 / (abs((new_speed - d_speed) + 1.99999)))

        delta_speed = (new_speed - d_speed) / d_speed
        if delta_speed < 0:
            cost += 0.7 * abs(delta_speed)
        elif new_speed > max_speed:
            cost += delta_speed

        # cost for illegal actions
        cost += 3.0 if not self._action_is_legal(agent, action) else 0.0

        if headway != -1:
            r_headway = math.log(headway / (self.headway_threshold * current_speed))
            cost += - cnf.HEADWAY_COST * r_headway if r_headway < 0 else 0

        if ttc != -1 and ttc < self.TTC_threshold:
            if action == a_dec:
                reward += 1.2
            elif action == a_left and tr_util.change_lane_chance(vehicleID=agent, change_direction=1):
                reward += 0.6

        reward += 0.3 if action == a_right and tr_util.change_lane_chance(vehicleID=agent, change_direction=-1) else 0

        return reward - cost

    def _update_state(self):
        """
        state(posX, posY, speed(m/s), edge_id, lane_indx, dist_merge_node, merge_node_visibility(0 or 1), TTC,
        headway, trip_time_delay)
        """
        def get_state(vehicle):
            posX = traci.vehicle.getPosition(vehicle)[0]
            posY = traci.vehicle.getPosition(vehicle)[1]
            sp = round(traci.vehicle.getSpeed(vehicle), 4)
            rID = self.s_edge_mapping[traci.vehicle.getRoadID(vehicle)]
            lane = traci.vehicle.getLaneIndex(vehicle)
            dis_m = round(tr_util.get_distance_to_merge_point(vehicle, self.merging_node), 2)
            mer_pnt_visib = self.s_mrg_visib_mapping[dis_m < self.observation_range]
            ttc = round(tr_util.ttc_with_ramp_veh(vehicle, self.on_ramp_edge), 2)
            headway = round(tr_util.headway_distance(vehicle), 2)
            trip_time_delay = round(tr_util.trip_time_delay(vehicle), 2)
            return [posX, posY, sp, rID, lane, dis_m, mer_pnt_visib, ttc, headway, trip_time_delay]

        states = []
        for veh in self.controlled_vehicles:
            if not self.agent_is_exist(veh):
                states.append(self.init_state_vals)
            elif self.agent_is_collide(veh):
                states.append(self.init_state_vals)
                self.remove_agent(veh)
                print(f"agent {veh},cause collision, removed from Env")
            elif self.agent_is_arrived(veh):
                states.append(get_state(veh))
                print(f"agent {veh},has arrived and removed from Env.")
                self.arrived_vehicles_state["vehicles"].append(veh)
                self.arrived_vehicles_state["terminal_state"].append(get_state(veh))
                self.remove_agent(veh)
            else:
                states.append(get_state(veh))

        self.state = states
        return self.state

    def _action(self, state, actions):
        for indx, (action, vehicle) in enumerate(zip(actions, self.controlled_vehicles)):
            if self.agent_is_exist(vehicle) and self._action_is_legal(vehicle, action):
                if state[indx][s_edge] == self.s_edge_mapping[self.on_ramp_edge]:
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

    @staticmethod
    def _act(vehID, action):
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

        # we want the environment to be dynamic
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

        # # hard braking for highway and ramp
        # if road_id == self.on_ramp_edge:
        #     if traci.vehicle.getAcceleration(vehicle) >= self.on_ramp_properties['hard_braking']:
        #         return False
        # else:
        #     if traci.vehicle.getAcceleration(vehicle) >= self.on_highway_properties['hard_braking']:
        #         return False

        return True

    @staticmethod
    def agent_is_exist(vehicle):
        return vehicle in traci.vehicle.getIDList()

    def agent_is_arrived(self, vehicle):
        return traci.vehicle.getRoadID(vehicle) == self.terminal_edge

    @staticmethod
    def agent_is_collide(vehicle):
        return vehicle in traci.simulation.getCollidingVehiclesIDList()

    @staticmethod
    def remove_agent(veh):
        traci.vehicle.remove(veh)

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
            # TODO : for the sync make sure the actor removed properly
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
