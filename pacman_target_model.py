import gym
import tensorflow as tf
import numpy as np
from collections import deque
import random

GAMMA = 0.99
LAMBDA = 0.9999
MAX_MEMORY_LEN = 100000

def preprocess(img):
    img = np.mean(img, axis=2).astype(np.uint8)
    img = img[1:176:2, ::2]
    img = img[:, :, np.newaxis]
    return img

def huber_loss(y_true, y_pred):
    error = y_true - y_pred
    quadratic_term = error*error / 2
    linear_term = abs(error) - 1/2
    use_linear_term = (abs(error) > 1.0)
    use_linear_term = tf.keras.backend.cast(use_linear_term, 'float32')
    return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term

class Brain:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.model = self.make_model()
        self.target_model = self.copy_model(self.model)

    def make_model(self):
        state_inp = tf.keras.layers.Input(self.observation_space)
        action_inp = tf.keras.layers.Input((self.action_space))
        x = tf.keras.layers.Lambda(lambda x:x/255)(state_inp)
        x = tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation="relu")(x)
        x = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation="relu")(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation="relu")(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        out = tf.keras.layers.Dense(self.action_space)(x)
        filtered_out = tf.keras.layers.multiply([out, action_inp])
        model = tf.keras.Model(inputs=[state_inp, action_inp], outputs=filtered_out)
        model.compile(loss=huber_loss, optimizer=tf.keras.optimizers.RMSprop(lr=0.0001, rho=0.95, epsilon=0.01))
        model.summary()
        return model

    def fit(self, x, y, batch_size):
        self.model.fit(x, y, batch_size=batch_size, verbose=0)

    def predict(self, x):
        return self.target_model.predict(x)

    def copy_model(self, model):
        tf.keras.models.save_model(model, "tmp.h5", include_optimizer=True, save_format="h5")
        new_model = tf.keras.models.load_model("tmp.h5")
        return new_model

class Agent:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.brain = Brain(observation_space, action_space)
        self.memory = deque(maxlen=MAX_MEMORY_LEN)
        self.epsilon = 1
        self.steps = 0

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            state = np.expand_dims(state, axis=0)
            actions = np.expand_dims(np.ones(self.action_space), axis=0)
            return np.argmax(self.brain.predict(((state, actions))))

    def observe(self, sample):
        self.memory.append(sample)
        self.steps += 1
        self.epsilon *= LAMBDA

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        batch = [list(i) for i in zip(*batch)] #transposing the dimensions of the batch
        start_states = np.array(batch[0])
        actions = np.array(batch[1])
        actions = tf.one_hot(actions, depth=self.action_space)
        rewards = np.array(batch[2])
        next_states = np.array(batch[3])
        is_terminal = batch[4]
        next_Q_values = self.brain.predict([next_states, np.ones(actions.shape)])
        next_Q_values[is_terminal] = 0
        Q_values = rewards + GAMMA * np.max(next_Q_values, axis=1)
        self.brain.fit([start_states, actions], actions * Q_values[:, None], batch_size=len(start_states))

env = gym.make("MsPacmanDeterministic-v4")
action_space = env.action_space.n
# observation_space = env.observation_space.shape
observation_space = (88, 80, 1)
agent = Agent(observation_space, action_space)
run = 0
total_steps = 0
for i in range(2000):
    step = 0
    state = env.reset()
    state = preprocess(state)
    while True:
        step += 1
        total_steps += 1
        if total_steps % 10000 == 0:
            agent.brain.target_model = agent.brain.copy_model(agent.brain.model)
        env.render()
        action = agent.act(state)
        state_, reward, terminal, info = env.step(action)
        state_ = preprocess(state_)
        agent.observe([state, action, reward, state_, terminal])
        state = state_
        if terminal:
            print("Run: " + str(run+1) + ", exploration: " + str(agent.epsilon))
            break
        agent.replay(32)
    run += 1
