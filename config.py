"""Runner var"""
RL_OPTION = "MAA2C"
TRAINING_STRATEGY = "concurrent"
MODEL_TYPE = "tensor"
MODEL_NAME = f"{RL_OPTION}_{TRAINING_STRATEGY}"

"""simulation related var"""
MAX_SIM_STEPS = 1000
GUI_DELAY = 100
WAITING_AGENT = 0.0001
DELTA_SEC = 0.05  # step length this should be same for both simulation (CARLA and SUMO)

"""training var"""
EPISODES = 2500
EPISODES_BEFORE_TRAIN = 10
EVAL_INTERVAL = 50
EVAL_EPISODES = 3
MEMORY_SIZE = 2000  # affect the memory usage
BATCH_SIZE = 200
UPDATE_TARGET_FREQ = 40
AGGREGATE_STATS_EVERY = 10
ROLL_OUT_STEPS = 20

"""define RL var"""
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99
EPSILON = 0.99
EPSILON_START = 0.99
EPSILON_END = 0.01
EPSILON_DECAY = 0.001
CRITIC_LOSS = "mse"
MAX_GRAD_NORM = 0.5
REWARD_DISCOUNTED_GAMMA = 0.97


"""define the constants for the reward function"""
TIME_PENALTY = 0.01
COLLISION_PENALTY = 100
MERGING_REWARD = 100  # divided by distance factor







