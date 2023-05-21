import argparse
import logging
import time

import glob
import os
import sys

import carla
import traci

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumo_integration.bridge_helper import BridgeHelper
from sumo_integration.carla_simulation import CarlaSimulation
from sumo_integration.constants import INVALID_ACTOR_ID
from sumo_integration.sumo_simulation import SumoSimulation




# carla config
CARLA_MAP = "Town04"
CARLA_SER_IP = "127.0.0.1"
CARLA_SER_PORT = 2000
DELTA_SEC = 0.05
CLIENT_ORDER = 1 #client order number for the co-simulation TraCI connection (default: 1)

# sumo config
SUMO_CFG_FILE = "map/Town04.sumocfg"
SUMO_SHOW_GUI = True
SUMO_SER_IP = None
SUMO_SER_PORT = None

# sync config
SYN_VEH_LIGHTS = False
SYN_VEH_COLOR = False
SYN_VEH_ALL = False # synchronize all vehicle properties (default: False)

TLS_MANAGER = "none" # choices=["none", "sumo", "carla"]


# Simulation Synchronization
# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
class SimulationSynchronization(object):
    """
    SimulationSynchronization class is responsible for the synchronization of sumo and carla
    simulations.
    """
    def __init__(
        self,
        sumo_simulation,
        carla_simulation,
        tls_manager="none",
        sync_vehicle_color=False,
        sync_vehicle_lights=False,
    ):
        self.sumo = sumo_simulation
        self.carla = carla_simulation
        
        self.tls_manager = tls_manager
        self.sync_vehicle_color = sync_vehicle_color
        self.sync_vehicle_lights = sync_vehicle_lights
        
        if tls_manager == "carla":
            self.sumo.switch_off_traffic_lights()
        elif tls_manager == "sumo":
            self.carla.switch_off_traffic_lights()
        
        # Mapped actor ids.
        self.sumo2carla_ids = {}  # Contains only actors controlled by sumo.
        self.carla2sumo_ids = {}  # Contains only actors controlled by carla.
        BridgeHelper.blueprint_library = self.carla.world.get_blueprint_library()
        BridgeHelper.offset = self.sumo.get_net_offset()
        # Configuring carla simulation in sync mode.
        settings = self.carla.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.carla.step_length
        self.carla.world.apply_settings(settings)
        traffic_manager = self.carla.client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)
    
    def tick(self):
        """
        Tick to simulation synchronization
        """
        # -----------------
        # sumo-->carla sync
        # -----------------
        self.sumo.tick()

        # Spawning new sumo actors in carla (i.e, not controlled by carla).
        sumo_spawned_actors = self.sumo.spawned_actors - set(
            self.carla2sumo_ids.values()
        )
        for sumo_actor_id in sumo_spawned_actors:
            self.sumo.subscribe(sumo_actor_id)
            sumo_actor = self.sumo.get_actor(sumo_actor_id)

            carla_blueprint = BridgeHelper.get_carla_blueprint(
                sumo_actor, self.sync_vehicle_color
            )
            if carla_blueprint is not None:
                carla_transform = BridgeHelper.get_carla_transform(
                    sumo_actor.transform, sumo_actor.extent
                )
                carla_actor_id = self.carla.spawn_actor(
                    carla_blueprint, carla_transform
                )
                if carla_actor_id != INVALID_ACTOR_ID:
                    self.sumo2carla_ids[sumo_actor_id] = carla_actor_id
            else:
                self.sumo.unsubscribe(sumo_actor_id)

        # Destroying sumo arrived actors in carla.
        for sumo_actor_id in self.sumo.destroyed_actors:
            if sumo_actor_id in self.sumo2carla_ids:
                self.carla.destroy_actor(self.sumo2carla_ids.pop(sumo_actor_id))

        # Updating sumo actors in carla.
        for sumo_actor_id in self.sumo2carla_ids:
            carla_actor_id = self.sumo2carla_ids[sumo_actor_id]

            sumo_actor = self.sumo.get_actor(sumo_actor_id)
            carla_actor = self.carla.get_actor(carla_actor_id)

            carla_transform = BridgeHelper.get_carla_transform(
                sumo_actor.transform, sumo_actor.extent
            )
            if self.sync_vehicle_lights:
                carla_lights = BridgeHelper.get_carla_lights_state(
                    carla_actor.get_light_state(), sumo_actor.signals
                )
            else:
                carla_lights = None

            self.carla.synchronize_vehicle(
                carla_actor_id, carla_transform, carla_lights
            )

        # Updates traffic lights in carla based on sumo information.
        if self.tls_manager == "sumo":
            common_landmarks = (
                self.sumo.traffic_light_ids & self.carla.traffic_light_ids
            )
            for landmark_id in common_landmarks:
                sumo_tl_state = self.sumo.get_traffic_light_state(landmark_id)
                carla_tl_state = BridgeHelper.get_carla_traffic_light_state(
                    sumo_tl_state
                )

                self.carla.synchronize_traffic_light(landmark_id, carla_tl_state)

        # -----------------
        # carla-->sumo sync
        # -----------------
        self.carla.tick()

        # Spawning new carla actors (not controlled by sumo)
        carla_spawned_actors = self.carla.spawned_actors - set(
            self.sumo2carla_ids.values()
        )
        for carla_actor_id in carla_spawned_actors:
            carla_actor = self.carla.get_actor(carla_actor_id)

            type_id = BridgeHelper.get_sumo_vtype(carla_actor)
            color = (
                carla_actor.attributes.get("color", None)
                if self.sync_vehicle_color
                else None
            )
            if type_id is not None:
                sumo_actor_id = self.sumo.spawn_actor(type_id, color)
                if sumo_actor_id != INVALID_ACTOR_ID:
                    self.carla2sumo_ids[carla_actor_id] = sumo_actor_id
                    self.sumo.subscribe(sumo_actor_id)

        # Destroying required carla actors in sumo.
        for carla_actor_id in self.carla.destroyed_actors:
            if carla_actor_id in self.carla2sumo_ids:
                self.sumo.destroy_actor(self.carla2sumo_ids.pop(carla_actor_id))

        # Updating carla actors in sumo.
        for carla_actor_id in self.carla2sumo_ids:
            sumo_actor_id = self.carla2sumo_ids[carla_actor_id]

            carla_actor = self.carla.get_actor(carla_actor_id)
            sumo_actor = self.sumo.get_actor(sumo_actor_id)

            sumo_transform = BridgeHelper.get_sumo_transform(
                carla_actor.get_transform(), carla_actor.bounding_box.extent
            )
            if self.sync_vehicle_lights:
                carla_lights = self.carla.get_actor_light_state(carla_actor_id)
                if carla_lights is not None:
                    sumo_lights = BridgeHelper.get_sumo_lights_state(
                        sumo_actor.signals, carla_lights
                    )
                else:
                    sumo_lights = None
            else:
                sumo_lights = None

            self.sumo.synchronize_vehicle(sumo_actor_id, sumo_transform, sumo_lights)

        # Updates traffic lights in sumo based on carla information.
        if self.tls_manager == "carla":
            common_landmarks = (
                self.sumo.traffic_light_ids & self.carla.traffic_light_ids
            )
            for landmark_id in common_landmarks:
                carla_tl_state = self.carla.get_traffic_light_state(landmark_id)
                sumo_tl_state = BridgeHelper.get_sumo_traffic_light_state(
                    carla_tl_state
                )

                # Updates all the sumo links related to this landmark.
                self.sumo.synchronize_traffic_light(landmark_id, sumo_tl_state)
    
    def close(self):
        """
        Cleans synchronization.
        """
        # Configuring carla simulation in async mode.
        settings = self.carla.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.carla.world.apply_settings(settings)

        # Destroying synchronized actors.
        for carla_actor_id in self.sumo2carla_ids.values():
            self.carla.destroy_actor(carla_actor_id)

        for sumo_actor_id in self.carla2sumo_ids.values():
            self.sumo.destroy_actor(sumo_actor_id)

        # Closing sumo and carla client.
        self.carla.close()
        self.sumo.close()



def _sync_loop(callback):
    
    """
    Entry point for sumo-carla co-simulation.
    """
    sumo_simulation = SumoSimulation(
        SUMO_CFG_FILE,
        DELTA_SEC,
        SUMO_SER_IP,
        SUMO_SER_PORT,
        SUMO_SHOW_GUI,
        CLIENT_ORDER,
    )
    
    carla_simulation = CarlaSimulation(
        CARLA_SER_IP, CARLA_SER_PORT, DELTA_SEC
    )
    
    synchronization = SimulationSynchronization(
        sumo_simulation,
        carla_simulation,
        TLS_MANAGER,
        SYN_VEH_COLOR,
        SYN_VEH_LIGHTS,
    )
    
    try:
        while True:
            start = time.time()
            
            synchronization.tick()
            
            #additional execution here
            
            end = time.time()
            elapsed = end - start
            if elapsed < DELTA_SEC:
                time.sleep(DELTA_SEC - elapsed)
        
    except KeyboardInterrupt:
        logging.info("Cancelled by user.")
    finally:
        logging.info("Cleaning synchronization")
        synchronization.close()

class TraCiSync():
    '''
    Synchronization carla&sumo, full contorl with TraCi
    use simulationStep(callBack) for advance one step and execute the callback function if need
    use colse() to close connection
    '''
    def __init__(self, client_order = CLIENT_ORDER):
        
        print("start SumoSimulation")
        self.sumo_simulation = SumoSimulation(
            SUMO_CFG_FILE,
            DELTA_SEC,
            SUMO_SER_IP,
            SUMO_SER_PORT,
            SUMO_SHOW_GUI,
            client_order,
        )
        
        print("start CarlaSimulation")
        self.carla_simulation = CarlaSimulation(
            CARLA_SER_IP, CARLA_SER_PORT, DELTA_SEC
        )
        
        print("start SimulationSynchronization")
        self.synchronization = SimulationSynchronization(
            self.sumo_simulation,
            self.carla_simulation,
            TLS_MANAGER,
            SYN_VEH_COLOR,
            SYN_VEH_LIGHTS,
        )
    
    def simulationStep(self, callback=None):
        start = time.time()
        self.synchronization.tick()
        #additional execution here
        if callback is not None:
            callback()
        end = time.time()
        elapsed = end - start
        if elapsed < DELTA_SEC:
            time.sleep(DELTA_SEC - elapsed)
    
    def close(self):
        self.synchronization.close()


def run():
    '''
    running synchronization carla&sumo running without any control
    load Map Carla/Maps/Town04
    load Cfg map/Town04.sumocfg
    Run map/Town04/rou/rou.xml
    '''
    
    try:
        client = carla.Client(CARLA_SER_IP, CARLA_SER_PORT)
        client.set_timeout(15.0)
        world = client.get_world()
        world_map_name = world.get_map().name.replace("Carla/Maps/","")
        if CARLA_MAP != world_map_name:
            print(f"load required {CARLA_MAP} map instead of {world.get_map().name}")
            client.load_world(CARLA_MAP)
        
        _sync_loop()
    except RuntimeError:
        print("failed to connect to carla server")
    except KeyboardInterrupt:
        logging.info("Cancelled by user.")
    except traci.exceptions.FatalTraCIError:
        logging.info("sumo close connection.")
    


if __name__ == "__main__":
    run()
