import traci

# Accelerate a vehicle
def accelerate(vehicle_id, acceleration=0.20):
    current_speed = traci.vehicle.getSpeed(vehicle_id)
    target_speed = current_speed + acceleration
    traci.vehicle.setSpeed(vehicle_id, target_speed)

# Decelerate a vehicle
def decelerate(vehicle_id, deceleration=0.37):
    current_speed = traci.vehicle.getSpeed(vehicle_id)
    target_speed = current_speed - deceleration
    traci.vehicle.setSpeed(vehicle_id, target_speed)

# def accelerate(vehicle,value = 15,duration = 5):
#     '''
#     slowDoww() function used to change the speed smoothly
#     '''
#     s = traci.vehicle.getSpeed(vehicle)
#     traci.vehicle.slowDown(vehicle, s + value, duration)

# def decelerate(vehicle,value = 10,duration = 1):
#     s = traci.vehicle.getSpeed(vehicle)
#     if s >= 0:
#         traci.vehicle.slowDown(vehicle, s - value, duration)

def change_to_left(vehicle):
    '''make sure vehicle can perform this lane change'''
    edge_id = traci.vehicle.getRoadID(vehicle)
    curr_lane = traci.vehicle.getLaneIndex(vehicle)
    to_lane = curr_lane + 1
    target_lane_id = f"{edge_id}_{to_lane}"
    traci.vehicle.changeLane(vehicle,to_lane,3)

def roadID_to_num(roadID, roadID_Mapping):
    '''
    roadID ex:'40.0.00', roadID_mapping ex: {'40.0' : 0 , ':720' : 1}
    roadID_to_num(roadID, roadID_Mapping) => number
    '''
    for prefix, value in roadID_Mapping.items():
        if roadID.startswith()(prefix):
            return value
        else:
            raise ValueError(f"roadID {roadID} does not included in roadID_Mapping")

def change_to_right(vehicle):
    '''make sure vehicle can perform this lane change'''
    edge_id = traci.vehicle.getRoadID(vehicle)
    curr_lane = traci.vehicle.getLaneIndex(vehicle)
    to_lane = curr_lane - 1 if curr_lane > 0 else curr_lane
    target_lane_id = f"{edge_id}_{to_lane}"
    traci.vehicle.changeLane(vehicle,to_lane,3)

def get_distance_to_merge_point(vehID, merge_node):
    '''
    distance to merging node (reeturn -1 if vehicle not in the simulation)
    get_distance_to_merge_point(str, str) -> float
    '''
    if vehID in traci.vehicle.getIDList():
        vehicle_pos = traci.vehicle.getPosition(vehID)
        jun_pos = traci.junction.getPosition(merge_node)
        return abs(traci.simulation.getDistance2D(vehicle_pos[0],vehicle_pos[1],jun_pos[0],jun_pos[1]))
    return -1


def get_closest_vehicle_on_ramp(on_highway_vehID, on_ramp_edge):
    '''
    get ID of closest vehicle on-ramp and its distance to ego vehicle on highway
    get_closest_vehicle_on_ramp(str, str) -> (str, float)
    '''
    # Get the position of the ego vehicle on the highway
    ego_position = traci.vehicle.getPosition(on_highway_vehID)
    
    # Get the positions and IDs of all controlled_vehicles on the on-ramp
    on_ramp_vehicle_ids = traci.edge.getLastStepVehicleIDs(on_ramp_edge)
    on_ramp_vehicle_positions = [traci.vehicle.getPosition(v_id) for v_id in on_ramp_vehicle_ids]
    
    on_ramp_vehicle_ids_positions = list(zip(on_ramp_vehicle_ids, on_ramp_vehicle_positions))
    if on_ramp_vehicle_ids_positions:
        # Find the closest vehicle on the on-ramp to the ego vehicle
        id, pos = min(on_ramp_vehicle_ids_positions, key=lambda pair: abs(pair[1][0] - ego_position[0]))
        # Calculate the distance
        # distance = l[1][0] - ego_position[0]
        return id, pos
    else:
        return None,None



def TTC_to_ramp_veh(vehID, on_ramp_edge):
    '''
    Compute the time to collision between the ego vehicle and the nearest vehicle on-ramp
    TTC_to_near_veh(str) -> float
    '''
    ego_speed = traci.vehicle.getSpeed(vehID)
    ego_lane = traci.vehicle.getLaneID(vehID)
    ego_position = traci.vehicle.getPosition(vehID)
    
    # Get the speeds of all controlled_vehicles on the on-ramp
    #on_ramp_vehicle_ids = traci.edge.getLastStepVehicleIDs(on_ramp_edge)
    #on_ramp_vehicle_speeds = [traci.vehicle.getSpeed(v_id) for v_id in on_ramp_vehicle_ids]
    
    clo_id, pos = get_closest_vehicle_on_ramp(vehID, on_ramp_edge)
    if clo_id:
        # Find the closest vehicle on the on-ramp to the ego vehicle
        #speed_diff = abs(traci.vehicle.getSpeed(clo_veh_id) - ego_speed)
        ttc = calculate_ttc(vehID, clo_id)
        # Calculate the time to collision (TTC)
    else:
        ttc = 9999
    return ttc

def calculate_ttc(veh1, veh2):
    
    if veh1 == veh2:
        return 9999
    # Get positions of the vehicles
    x1, y1 = traci.vehicle.getPosition(veh1)
    x2, y2 = traci.vehicle.getPosition(veh2)
    
    # Calculate relative position vector
    rel_pos_x = x2 - x1
    rel_pos_y = y2 - y1
    
    # Get velocities of the vehicles
    v1 = traci.vehicle.getSpeed(veh1)
    v2 = traci.vehicle.getSpeed(veh2)
    
    # Calculate relative velocity
    rel_vel_x = v2 - v1
    rel_vel_y = 0  # Assume vehicles are moving along the same axis
    
    # Calculate dot product of relative position and relative velocity
    dot_product = rel_pos_x * rel_vel_x + rel_pos_y * rel_vel_y
    
    # Check if the vehicles are moving towards each other
    if dot_product < 0:
        # Calculate the TTC
        ttc = -dot_product / (rel_vel_x * rel_vel_x + rel_vel_y * rel_vel_y)
        return ttc
    
    # If the vehicles are moving away or have the same speed, consider TTC as infinite
    return 9999

def headway_distance(vehicle):
    # Get the vehicle's position and lane index
    vehicle_position = traci.vehicle.getPosition(vehicle)
    vehicle_lane_index = traci.vehicle.getLaneIndex(vehicle)
    
    # Get the leader vehicle's position and lane index
    leader_vehicle = traci.vehicle.getLeader(vehicle,100)
    if leader_vehicle:
        headway_distance = leader_vehicle[1]
    else:
        headway_distance = 9999
    return headway_distance

def change_lane_chance(vehicleID, change_diraction):
    '''
    Return whether the vehicle could change lanes in the specified direction 
    (right: -1, left: 1. sublane-change within current lane: 0).
    Return the lane change state for the vehicle.
    '''
    return traci.vehicle.getLaneChangeState(vehicleID, change_diraction)


def trip_time(vehID, arrival_edge):
    speed = traci.vehicle.getSpeed(vehID)
    distance = traci.vehicle.getDistance(vehID)
    roadID = traci.vehicle.getRoadID(vehID)
    sim_time = traci.simulation.getTime()
    # Calculate trip time
    if speed > 0 and roadID == arrival_edge:
        trip_time = (distance / speed) + sim_time
    else:
        trip_time = 9999 # Vehicle is stationary, set trip time to infinity
    
    return trip_time