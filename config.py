
'''
# silulation related var
'''

MAX_SIM_STEPS = 150
GUI_DELAY = 100
WHAITING_AGENT = 0.0001
DELTA_SEC = 0.05 #step length this should be same for both simulation (CARLA and SUMO)
SHOW_EVERY = 10 #for render only EPISODES % SHOW_EVERY times

'''training var'''
EPISODES = 500
BATCH_SIZE= 50
UPDATE_TARGET_FREQ = 20
SAVE_MODEL_EVERY = 20
MEMORY_SIZE=100 # affect the memory usage
AGGREGATE_STATS_EVERY = 10


'''
# define the constants for the reward function
'''

TIME_PENALTY = 0.01
COLLISION_PENALTY = 100
MERGING_REWARD = 100 # devided by distance factor


'''
# define RL var
'''
LEARNING_RATE=0.001
DISCOUNT_FACTOR=0.99
EPSILON=1.0
EPSILON_MIN= 0.01
EPSILON_DECAY=0.995


EGO_VEHICLE_TYPE = "ego_vehicle"
MODEL_NAME = "MADQN"