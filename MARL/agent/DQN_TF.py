import keras.models as models
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque
from util.ModifiedTensorBoard import ModifiedTensorBoard

seed = 10
np.random.seed(seed)
random.seed(seed)


class DQN_TF:
    """
    DQN agent
    using tensorflow model approximation based method
    - take exploration action, expect epsilon value or use decay_epsilon()
    - save experiences to replay memory
    - train model and update values on batch sample
    - save model
    @mahmoudtaouti
    """
    def __init__(self, state_dim, action_dim,
                 memory_capacity=10000, batch_size=100,
                 reward_gamma=0.99, reward_scale=1.,
                 target_update_freq=30,
                 actor_hidden_size=128, actor_lr=0.001,
                 optimizer_type="rmsprop",
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=0.001,
                 load_model=None):

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale

        self.actor_hidden_size = actor_hidden_size
        self.actor_lr = actor_lr
        self.optimizer_type = optimizer_type
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # params for epsilon greedy
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.memory = deque(maxlen=memory_capacity)

        self.loaded_model = load_model

        self.actor = self._load_model() if load_model else self._build_model()
        self.target = self._build_model()

        self.target.set_weights(self.actor.get_weights())

    # Used to count when to update target network with main network's weights
    def _build_model(self):
        # neural network for deep-q learning model
        model = models.Sequential()
        model.add(Dense(self.actor_hidden_size, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(self.action_dim, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.actor_lr), metrics=['accuracy'])
        return model

    def _load_model(self):
        return models.load_model(self.loaded_model)

    def learn(self):

        # Start training only if certain number of samples is already saved
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        # Get current states from minibatch, then query NN model for Q values
        states = np.array([transition[0] for transition in batch])
        qs_list = self.actor.predict(states)

        # Get future states from batch, then query NN model for Q values
        new_states = np.array([transition[3] for transition in batch])
        future_qs_list = self.target.predict(new_states)

        # train the model on the batch of experiences
        for index, (state, action, reward, next_state, done) in enumerate(batch):
            # compute target q by: r + gamma * max_a { V(s_{t+1}) }
            new_q = self.reward_scale * reward + self.reward_gamma * max(future_qs_list[index]) * (1 - int(done))

            # Update Q value for given state
            current_qs = qs_list[index]
            current_qs[action] = new_q
            qs_list[index] = current_qs

        X = np.array(states)
        Y = np.array(qs_list)
        self.actor.fit(X, Y, batch_size=self.batch_size, epochs=3, verbose=0, shuffle=False)

    def shared_learning(self, n_agents, agent_index, shared_batch_sample):
        self.learn()

    def update_target(self):
        self.target.set_weights(self.actor.get_weights())

    # predict or get random action
    def exploration_action(self, state, epsilon=None):

        self.epsilon = epsilon if epsilon else self.epsilon
        if np.random.rand() > self.epsilon:
            return self.action(state)
        else:
            return random.randrange(0, self.action_dim)

    def action(self, state):
        X = np.array([state])
        qs = self.actor.predict(X)
        return np.argmax(qs)

    # Adds step's experience to a memory replay
    # (observation space, action, reward, new observation space, done)
    def remember(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def decay_epsilon(self):
        # decrease the exploration rate epsilon over time
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_end, self.epsilon)

    def save(self, global_step=None, out_dir='/model'):
        actor_file_path = out_dir + f"/actor_eps{global_step}.model"
        self.actor.save(filepath=actor_file_path)
