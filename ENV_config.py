"""define the constants for the reward function"""
COLLISION_COST = 100
HIGH_SPEED_REWARD = 1
HEADWAY_COST = 1
HEADWAY_TIME = 1.2
MERGING_LANE_REWARD = 4

"""simulation related var"""
MAX_SIM_STEPS = 1000
GUI_DELAY = 100
WAITING_AGENT = 0.0001
DELTA_SEC = 0.05  # step length this should be same for both simulation (CARLA and SUMO)


"""State var"""
REFERENCE_TRIP_DELAY = 20  # reference to maximum trip time delay for all vehicles could take
POSITION_RANGE = (0, 800)
SPEED_RANGE = (0, 50)
DISTANCE_RANGE = (0, 300)
TTC_RANGE = (0, 200)
HEADWAY_RANGE = (0, 200)
TRIP_DELAY_RANGE = (0, 50)


