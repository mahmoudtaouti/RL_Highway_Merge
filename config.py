"""Runner var"""
RL_OPTION = "MAA2C"
TRAINING_STRATEGY = "concurrent"
MODEL_TYPE = "torch"
MODEL_NAME = f"{RL_OPTION}_{TRAINING_STRATEGY}"

"""simulation related var"""
MAX_SIM_STEPS = 1000
GUI_DELAY = 100
WAITING_AGENT = 0.0001
DELTA_SEC = 0.05  # step length this should be same for both simulation (CARLA and SUMO)

"""training var"""
EPISODES = 5000
EPISODES_BEFORE_TRAIN = 5
EVAL_INTERVAL = 50
EVAL_EPISODES = 3
MEMORY_SIZE = 600  # affect the memory usage
BATCH_SIZE = 600
UPDATE_TARGET_FREQ = 40
AGGREGATE_STATS_EVERY = 10
ROLL_OUT_STEPS = 20

"""define RL var"""
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99
EPSILON = 0.99
EPSILON_START = 0.99
EPSILON_END = 0.01
EPSILON_DECAY = 0.0007
EPSILON_DECAY_METHOD = "exponential"
CRITIC_LOSS = "huber"
OPTIMIZER_TYPE = "rmsprop"
MAX_GRAD_NORM = 0.5
REWARD_DISCOUNTED_GAMMA = 0.98
ENTROPY_REG = 0.01


"""models var"""
ACTOR_HIDDEN_SIZE = 128
CRITIC_HIDDEN_SIZE = 128


"""define the constants for the reward function"""
COLLISION_COST = 100
HIGH_SPEED_REWARD = 1
HEADWAY_COST = 4
HEADWAY_TIME = 1.2
MERGING_LANE_REWARD = 4







