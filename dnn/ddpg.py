### Deep Deterministic Policy Gradient (DDPG)

import collections
import io
import math
import random

import numpy as np

from keras import layers
from keras.models import Model
from keras import optimizers
from keras import backend as K

import gym


class ReplayBuffer:
  def __init__(self, buf_size):
    self.buffer = collections.deque(maxlen=buf_size)
    self.experience = collections.namedtuple("Experience",
                                             field_names=["s0", "a", "r", "s1", "d"])

  def add(self, state, action, reward, next_state, d=None):
    e = self.experience(state, action, reward, next_state, d)
    self.buffer.append(e)

  def sample(self, batch_size=64):
    return random.sample(self.buffer, k=batch_size)

  def __len__(self):
    return len(self.buffer)


class OUNoise:
  def __init__(self, size, mu=0., theta=0.05, sigma=0.25):
    self.size = size
    self.mu = mu
    self.theta = theta
    self.sigma = sigma
    self.reset()

  def reset(self):
    self.state = np.ones(self.size)*self.mu

  def sample(self):
    x = self.state
    dx = self.theta * (self.mu - x) + self.sigma*np.random.randn(len(x))
    self.state = x + dx
    return self.state


class Actor:
  def policy_optimization_loss(action_gradients):
    def loss(y_true, y_pred):
      return -K.mean(action_gradients * y_pred)
    return loss

  def __init__(self, nstate, naction, units=[40,20], lr=0.0001):
    self.nstate = nstate
    self.naction = naction
    self.units = units
    self.lr = lr

    self.model = self.build_model()
    self.target = self.build_model()
    self.target.set_weights(self.model.get_weights())

  def build_model(self, target=False):
    states = layers.Input(shape=(self.nstate,), name='states')
    action_gradients = layers.Input(shape=(self.naction,))

    net = layers.Dense(units=self.units[0], activation='relu')(states)
    for nn in self.units[1:]:
      net = layers.Dense(units=nn, activation='relu')(net)
    actions = layers.Dense(units=self.naction, activation='tanh', name='actions')(net)

    model = Model(inputs=[states, action_gradients], outputs=actions)
    adam = optimizers.Adam(lr=self.lr)
    model.compile(optimizer=adam, loss=[Actor.policy_optimization_loss(action_gradients)])
    return model

  def predict(self, state):
    dummy = np.zeros(shape=(len(state), self.naction))
    return self.model.predict([state, dummy])

  def target_predict(self, states):
    dummy = np.zeros(shape=(len(states), self.naction))
    return self.target.predict_on_batch([states, dummy])

  def soft_update(self, tau):
    weights = np.array(self.model.get_weights())
    target_weights = np.array(self.target.get_weights())

    next_weights = tau * weights + (1. - tau) * target_weights
    self.target.set_weights(next_weights)

  def fit(self, states, actions, action_gradients):
    self.model.fit([states, action_gradients], [actions], verbose=0)


class Critic:

  def __init__(self, nstate, naction, units=[20,20], lr=0.05):
    self.nstate = nstate
    self.naction = naction
    self.units = units
    self.lr = lr

    self.model = self.build_model()
    self.add_method(self.model)
    self.target = self.build_model()
    self.target.set_weights(self.model.get_weights())

  def build_model(self):
    states = layers.Input(shape=(self.nstate,), name='states')
    actions = layers.Input(shape=(self.naction,), name='actions')

    inputs = layers.Concatenate()([states, actions])
    net = layers.Dense(units=self.units[0], activation='relu')(inputs)
    for nn in self.units[1:]:
      net = layers.Dense(units=nn, activation='relu')(net)
    Q_values = layers.Dense(units=1,name='Q_values')(net)

    model = Model(inputs=[states, actions], outputs=Q_values)
    optimizer = optimizers.Adam(lr=self.lr)
    model.compile(optimizer=optimizer, loss='mse')
    return model

  def add_method(self, model):
    action_gradients = K.gradients(model.output, model.input[1])
    self.get_action_gradients = K.function(inputs=[*model.input, K.learning_phase()],
                                           outputs=action_gradients)

  def train_on_batch(self, states, actions, Q_targets):
    self.model.train_on_batch(x=[states, actions], y=Q_targets)

  def target_predict_on_batch(self, states, actions):
    return self.target.predict_on_batch([states, actions])

  def soft_update(self, tau):
    weights = np.array(self.model.get_weights())
    target_weights = np.array(self.target.get_weights())

    next_weights = tau * weights + (1. - tau) * target_weights
    self.target.set_weights(next_weights)


class DDPG():
  def __init__(self, nstate=None, naction=None, bounds=None,
               mu=0., theta=0.05, sigma=0.25,
               buf_size=10000, batch_size=128,
               offset_scale=0.2, gamma=0.999, tau_actor=0.1, tau_critic=0.2,
               actor_units=[40,20], actor_lr=0.0001,
               critic_units=[20,20], critic_lr=0.05):
    self.nstate = nstate
    self.naction = naction
    self.bounds = np.array(bounds)

    self.actor = Actor(self.nstate, self.naction, units=actor_units, lr=actor_lr)
    self.critic = Critic(self.nstate, self.naction, units=critic_units, lr=critic_lr)

    self.exploration_mu = mu
    self.exploration_theta = theta
    self.exploration_sigma = sigma
    self.noise = OUNoise(self.naction, self.exploration_mu,
                         self.exploration_theta, self.exploration_sigma)

    self.buffer_size = buf_size
    self.batch_size = batch_size
    self.memory = ReplayBuffer(self.buffer_size)

    self.offset_scale = offset_scale
    self.gamma = gamma
    self.tau_actor = tau_actor
    self.tau_critic = tau_critic

  def reset_episode(self, state):
    self.noise.reset()
    self.last_state = state

  def step(self, action, reward, next_state, done):
    self.memory.add(self.last_state, action, reward, next_state, done)

    if len(self.memory) > self.batch_size:
      experiences = self.memory.sample(self.batch_size)
      self.learn(experiences)

    self.last_state = next_state

  def act(self, state):
    state = np.reshape(state, [-1, self.nstate])

    true_action = self.actor.predict(state)[0]
    noise = self.noise.sample()      
    action = np.clip(true_action*self.offset_scale + noise, -1., 1.)
    lo = self.bounds[:, 0]
    hi = self.bounds[:, 1]
    scale = (hi - lo)*0.5
    shift = scale + lo
    action = (action + shift)*scale
    true_action = (true_action + shift)*scale
    return action, true_action

  def learn(self, experiences):
    def to_list(array, key):
      return [getattr(x, key) for x in array if x is not None]

    states = np.vstack(to_list(experiences, 's0'))
    actions = np.array(to_list(experiences, 'a')).astype(np.float32).reshape(-1, self.naction)
    rewards = np.array(to_list(experiences, 'r')).astype(np.float32).reshape(-1, 1)
    next_states = np.vstack(to_list(experiences, 's1'))
    dones = np.array(to_list(experiences, 'd')).astype(np.uint8).reshape(-1, 1)

    actions_next = self.actor.target_predict(next_states)
    Q_targets_next = self.critic.target_predict_on_batch(next_states, actions_next)

    Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
    self.critic.train_on_batch(states, actions, Q_targets)

    learning = 0
    action_gradients = np.reshape(self.critic.get_action_gradients([states, actions, learning]),
                                  (-1, self.naction))
    self.actor.fit(states, actions, action_gradients)

    self.critic.soft_update(self.tau_critic)
    self.actor.soft_update(self.tau_actor)


class Normaliser:
  def __init__(self):
    self.n_sample = 0
    self.average = 0.
    self.M2 = 0.

  def update(self, x):
    self.n_sample += 1
    delta = x - self.average
    self.average += delta / self.n_sample
    delta2 = x - self.average
    self.M2 += delta * delta2

  def n_sample(self):
    return self.n_sample

  def mean(self):
    return self.average

  def variance(self):
    return self.M2/self.n_sample if self.n_sample > 1 else 0.

  def sigma(self):
    return math.sqrt(self.variance())

  def standardise(self, x):
    rc = (x - self.average)/(5*self.sigma()) if self.sigma() else (x - self.average)
    return np.clip(rc, -1., 1.)

  def standardise_with_update(self, x):
    self.update(x)
    return self.standardise(x)
    

class GymContinuous():

  def __init__(self, model, pars):
    env = self.new_env(model)
    self.naction = env.action_space.shape[0]
    self.nstate = env.observation_space.shape[0]
    self.bounds_actions = list(zip(env.action_space.low, env.action_space.high))
    self.bounds_states = list(zip(env.observation_space.low, env.observation_space.high)) 
    self.env = env
    self.model = model

    self.normalisers = np.zeros(self.nstate, dtype=object)
    for ii in range(self.nstate):
      self.normalisers[ii] = Normaliser()

    pars['nstate'] = self.nstate
    pars['naction'] = self.naction
    pars['bounds'] = self.bounds_actions

    self.agent = DDPG(**pars)
    self.epoch = collections.namedtuple("epoch",
                   field_names=["sum_reward", "done", "action_mean", "action_stddev", "steps"])

  def new_env(self, model):
    return gym.make(model).unwrapped

  def normalise_state(self, state):
    s = np.array(state)
    for ii in range(self.nstate):
      s[ii] = self.normalisers[ii].standardise_with_update(state[ii])
    return s

  def run_epoch(self, max_steps, render=False, training=True):
    state = self.normalise_state(self.env.reset())
    self.agent.reset_episode(state)
    actions_list = []
    sum_reward = 0
    steps = 0

    while steps < max_steps:
      steps += 1
      noisy_action, true_action = self.agent.act(state)
      action = noisy_action if training else true_action

      next_state, reward, done, info = self.env.step(action)
      next_state = self.normalise_state(next_state)
      state = next_state
      sum_reward += reward
      actions_list.append(true_action)

      if training: self.agent.step(action, reward, next_state, done)
      if render: self.env.render()
      if done: break

    action_mean = np.mean(actions_list)
    action_stddev = np.std(actions_list)

    epoch = self.epoch(sum_reward, done, action_mean, action_stddev, steps)
    return epoch

  def run_model(self, max_epochs=100, n_solved=1, r_solved=90,
                max_steps=1000, verbose=1, render=False, training=True):
    solved = False
    train_hist = collections.deque(maxlen=n_solved)
    test_hist = collections.deque(maxlen=n_solved)

    for epoch in range(1, max_epochs+1):
      train_epoch = self.run_epoch(max_steps=max_steps, training=training)
      test_epoch = self.run_epoch(max_steps=max_steps, training=False, render=render)

      train_hist.append([train_epoch.sum_reward, train_epoch.steps])
      test_hist.append([test_epoch.sum_reward, test_epoch.steps])

      train_running = np.mean([r for r, s in train_hist])
      test_running = np.mean([r for r, s in test_hist])

      print_vals = np.array([epoch,
                             train_epoch.sum_reward, train_epoch.steps, train_running,
                             test_epoch.sum_reward, test_epoch.steps, test_running])

      self.print_epoch(print_vals, verbose)
      if test_running > r_solved and epoch > n_solved:
        print('\nSolved! Average of {:4.1f} from episode {:3d}'
              ' to {:3d}'.format(test_running, epoch -
                                        n_solved + 1, epoch))
        solved = epoch
        break

    return train_hist, test_hist, solved

  def print_epoch(self, vals, verbose):
    if verbose == 1:
      pstr = ('Epoch:{:4.0f}\nTrain: reward:{: 6.1f} steps:{:6.0f} running mean:{: 6.1f} \n'
              'Test:  reward:{: 6.1f} steps:{:6.0f} running mean:{: 6.1f}\n'.format(*vals))
    elif verbose == 0:
      pstr = ('Epoch {:4.0f} train reward:{:6.1f} steps:{:6.0f} '
              'test reward:{:6.1f} steps:{:6.0f}\r'.format(*vals[[0, 1, 2, 4, 5]]))
    elif verbose == -1:
      return

    print(pstr)


if __name__ == '__main__':
  pars = {
    'mu': 0.,
    'theta': 0.05,
    'sigma': 0.30,
    'buf_size': 10000,
    'batch_size': 128,
    'offset_scale': 0.2,
    'gamma': 0.999,
    'tau_actor': 0.05,
    'tau_critic': 0.05,
    'actor_units': [80,40], 
    'actor_lr': 0.0001,
    'critic_units': [40,40],
    'critic_lr': 0.01,
  }


  env = 'Pendulum-v0'
  o = GymContinuous(env, pars)
  o.run_model(max_epochs=200, n_solved=20, r_solved=-150)
  o.run_model(max_epochs=1, n_solved=0, r_solved=-150, verbose=0, render=True, training=False)

# eof
