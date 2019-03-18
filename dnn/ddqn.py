import random

import collections

import numpy as np

import keras
from keras import layers
from keras.models import Model
from keras import backend as K
from keras import optimizers

import tensorflow as tf


""" double-DQN """
class DDQN:
  def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred

    condition = K.abs(error) < delta
    square = 0.5*K.square(error)
    linear = delta*(K.abs(error) - 0.5*delta)

    loss = tf.where(condition, linear, square)
    return K.mean(loss)

  def __init__(self, nstate, naction):
    self.nstate = nstate
    self.naction = naction
    self.lr = 1.e-3

    self.model = self.build_model()
    self.target = self.build_model() 
    self.soft_update(1.)    # copy model weights to target

  def build_model(self):
    states = layers.Input(shape=(self.nstate,))

    ## value network
    net = layers.Dense(16)(states)
    net = layers.Activation("relu")(net)
    net = layers.Dense(16)(net)
    net = layers.Activation("relu")(net)
    net = layers.Dense(self.naction)(net)
    actions = layers.Activation("linear")(net)

    model = Model(inputs=states, outputs=actions)
    optimizer = optimizers.Adam(lr=self.lr)
    model.compile(optimizer=optimizer, loss=DDQN.huber_loss)

    return model

  def train(self, x, y, batch_size=32, verbose=0):
    self.model.fit(x, y, batch_size=batch_size, verbose=verbose)

  def predict(self, state, target=False):
    if len(state.shape) == 1:
      state = np.expand_dims(state, axis=0)

    rc = None
    if target:
      rc = self.target.predict(state)
    else:
      rc = self.model.predict(state)
    return rc

  def soft_update(self, tau=0.2):
    weights = np.array(self.model.get_weights())
    target_weights = np.array(self.target.get_weights())

    next_weights = tau * weights + (1. - tau) * target_weights
    self.target.set_weights(next_weights)


class ReplayBuffer:
  def __init__(self, buf_size):
    self.buf_size = buf_size
    self.buffer = collections.deque(maxlen=buf_size)
    self.experience = collections.namedtuple("Experience",
                                             field_names=["s0", "a", "r", "s1", "d"])

  def add(self, state, action, reward, next_state, d=None):
    e = self.experience(state, action, reward, next_state, d)
    self.buffer.append(e)

  def sample(self, batch_size=32):
    return random.sample(self.buffer, k=batch_size)

  def __len__(self):
    return len(self.buffer)


class Agent:
  steps = 0
  min_epsilon = 0.01
  scale = 1. - min_epsilon
  Lambda = 0.001
  update_target_frequency = 1000

  def __init__(self, nstate, naction, buf_size=10000):
    self.nstate = nstate
    self.naction = naction
    self.buf_size = buf_size
    self.epsilon = 1.0

    self.ddqn = DDQN(nstate, naction)
    self.memory = ReplayBuffer(buf_size)
    self.gamma = 0.99
    
  def act(self, state, training=True):
    rc = None
    if training and random.random() < self.epsilon:
      rc = random.randint(0, self.naction-1)
    else:
      rc = np.argmax(self.ddqn.predict(state))

    self.steps += 1
    self.epsilon = Agent.min_epsilon + Agent.scale * np.exp(-Agent.Lambda * self.steps)
    return rc

  def fit(self, batch_size=32):
    def to_list(array, key):
      return [getattr(x, key) for x in array]

    experiences = self.memory.sample(batch_size)
    states = np.array(to_list(experiences, 's0'))
    next_states = np.array(to_list(experiences, 's1'))
    actions = np.array(to_list(experiences, 'a'))
    rewards = np.array(to_list(experiences, 'r'))
    dones = np.array(to_list(experiences, 'd'))

    p0 = np.squeeze(self.ddqn.predict(states))
    p1 = np.squeeze(self.ddqn.predict(next_states, target=True))
    p2 = np.squeeze(self.ddqn.predict(next_states))

    y = np.array(p0)
    mask_y = keras.utils.to_categorical(actions, num_classes=self.naction).astype(np.bool)
    mask_p1 = keras.utils.to_categorical(np.argmax(p2, axis=1), num_classes=self.naction).astype(np.bool)
    y[mask_y.tolist()] = rewards + self.gamma * p1[mask_p1.tolist()] * (1 - dones)
    self.ddqn.train(states, y, batch_size=batch_size)
    self.ddqn.soft_update(0.1)


def run_episode(env, agent, render=False, training=True):
  done = False
  batch_size = 128
  total_reward = 0 

  s0 = env.reset()
  while not done:            
    a = agent.act(s0, training)
    s1, r, done, info = env.step(a)
    if render: env.render()

    total_reward += r
    agent.memory.add(s0, a, r, s1, done)
    if len(agent.memory) > batch_size:
      agent.fit(batch_size)

    s0 = s1
    run_episode.counter += 1
    total_reward += r

    if done: break

  return total_reward


if __name__ == '__main__':
  import gym

  model = 'CartPole-v0'
  maxlen = 100
  min_average = 195

  env = gym.make(model)
  nstate  = env.env.observation_space.shape[0]
  naction = env.env.action_space.n

  agent = Agent(nstate, naction)

  run_episode.counter = 0
  rb = collections.deque(maxlen=maxlen)
  counter = 0
  while True:
    r = run_episode(env, agent, counter%maxlen==0)
    rb.append(r)
    counter += 1
    average = np.mean(rb)
    if counter % (maxlen/10) == 0:
      print('epoch = {:4d} running mean = {:6.2f}'.format(counter, average))

    if counter > maxlen-1 and average > min_average:
      print('done ', counter, average)
      break

  reward = run_episode(env, agent, render=True, training=False)
  print(reward)

# eof
