import random

import keras
from keras import layers
from keras.models import Model
from keras import backend as K
from keras import optimizers

import tensorflow as tf

from PIL import Image

import random
import collections
import numpy as np


""" ref: https://github.com/jaromiru/AI-blog/blob/master/SumTree.py """
class SumTree:

  def __init__(self, buf_size):
    self.buf_size = buf_size
    self.tree = np.zeros(2*buf_size - 1)
    self.data = np.zeros(buf_size, dtype=object)
    self.index = 0
    self.full = False

  def add(self, p, data):
    idx = self.index + self.buf_size - 1

    self.data[self.index] = data
    self.update(idx, p)

    self.index += 1
    if self.index >= self.buf_size:
      self.index = 0
      self.full = True

  def _propagate(self, idx, change):
    parent = (idx - 1) // 2
    self.tree[parent] += change
    if parent != 0: self._propagate(parent, change)

  def update(self, idx, p):
    change = p - self.tree[idx]
    self.tree[idx] = p
    self._propagate(idx, change)

  def _retrieve(self, idx, s):
    left = 2*idx + 1
    right = left + 1

    if left >= len(self.tree):
      return idx

    if s <= self.tree[left]:
      return self._retrieve(left, s)
    else:
      return self._retrieve(right, s-self.tree[left])

  def get(self, s):
    idx = self._retrieve(0, s)
    data = idx - self.buf_size + 1
    return (idx, self.tree[idx], self.data[data])

  def sum_priorities(self):
    return self.tree[0]

  def __len__(self):
    rc = self.buf_size
    if not self.full:
      rc = self.index
    return rc


class PERBuffer:

  def __init__(self, buf_size, epsilon=0.01, alpha=0.6):
    self.tree = SumTree(buf_size)
    self.experience = collections.namedtuple("Experience",
                                    field_names=["s0", "a", "r", "s1", "d"])
    self.buf_size = buf_size
    self.epsilon = epsilon
    self.alpha = alpha

  def _getPriority(self, error):
    rc = (error + self.epsilon)**self.alpha
    return rc

  def update(self, idx, error):
    p = self._getPriority(error)
    for ii in range(len(idx)):
      self.tree.update(idx[ii], p[ii])

  def add(self, error, state, action, reward, next_state, done=None):
    p = self._getPriority(error)
    e = self.experience(state, action, reward, next_state, done)
    self.tree.add(p, e)

  def sample(self, batch_size=64):
    segment = self.tree.sum_priorities() / batch_size
    idxs = np.zeros(batch_size, dtype=np.int32)
    batch = np.zeros(batch_size, dtype=object)
    priorities = np.zeros(batch_size, dtype=object)

    for ii in range(batch_size):
      a = segment * ii
      b = segment * (ii + 1)
      s = random.uniform(a, b)
      (idx, p, data) = self.tree.get(s)
      idxs[ii] = idx
      batch[ii] = data
      priorities[ii] = p

    return idxs, batch, priorities

  def minimum_priority(self):
    ma = np.ma.masked_equal(self.tree.tree[-self.buf_size:], 0., copy=False)
    return ma.min()

  def sum_priorities(self):
    return self.tree.sum_priorities()

  def __len__(self):
    return len(self.tree)


""" double-DQN with PER """
class DDQN_PER:
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
    model.compile(optimizer=optimizer, loss=DDQN_PER.huber_loss)

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

    self.ddqn_per = DDQN_PER(nstate, naction)
    self.memory = PERBuffer(buf_size)
    self.gamma = 0.99
    
  def act(self, state, training=True):
    rc = None
    if training and random.random() < self.epsilon:
      rc = random.randint(0, self.naction-1)
    else:
      rc = np.argmax(self.ddqn_per.predict(state))

    self.steps += 1
    self.epsilon = Agent.min_epsilon + Agent.scale * np.exp(-Agent.Lambda * self.steps)
    return rc

  def get_targets(self, states, actions, rewards, next_states, dones):
    p0 = np.squeeze(self.ddqn_per.predict(states))
    p1 = np.squeeze(self.ddqn_per.predict(next_states, target=True))
    p2 = np.squeeze(self.ddqn_per.predict(next_states))

    y = np.array(p0)
    mask_p0 = keras.utils.to_categorical(actions, num_classes=self.naction).astype(np.bool)
    if len(p2.shape) == 1:
      mask_p1 = keras.utils.to_categorical(np.argmax(p2,axis=0), num_classes=self.naction).astype(np.bool)
    else:
      mask_p1 = keras.utils.to_categorical(np.argmax(p2, axis=1), num_classes=self.naction).astype(np.bool)
    y[mask_p0.tolist()] = rewards + self.gamma * p1[mask_p1.tolist()] * (1 - dones)
    errors = abs(p0[mask_p0.tolist()] - y[mask_p0.tolist()])
    return states, y, errors

  def fit(self, batch_size=32):
    def to_list(array, key):
      return [getattr(x, key) for x in array if type(x) is not type(self.memory.experience)]

    idxs, experiences, priorities = self.memory.sample(batch_size)

    states = np.array(to_list(experiences, 's0'))
    next_states = np.array(to_list(experiences, 's1'))
    actions = np.array(to_list(experiences, 'a'))
    rewards = np.array(to_list(experiences, 'r'))
    dones = np.array(to_list(experiences, 'd'))

    states, y, errors = self.get_targets(states, actions, rewards, next_states, dones)
    self.memory.update(idxs, errors)

    self.ddqn_per.train(states, y, batch_size=batch_size)
    self.ddqn_per.soft_update(0.1)


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
    x, y, error = agent.get_targets(s0, a, r, s1, done)
    agent.memory.add(error, s0, a, r, s1, done)
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
    render = (counter % maxlen == 0)
    reward = run_episode(env, agent, render)
    rb.append(reward)
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
