import gym
import random
from collections import namedtuple, deque

import numpy as np

from keras.layers import Input, Dense, Add
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K


class Actor:
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size

    self.build_model()
    self.build_model(True)
    self.target.set_weights(self.model.get_weights())

  def build_model(self, target=False):
    states = Input(shape=(self.state_size,), name='states')
    net = Dense(units=40, activation='relu')(states)
    net = Dense(units=20, activation='relu')(net)
    actions = Dense(units=self.action_size, activation='tanh', name='actions')(net)

    if target:
      self.target = Model(inputs=states, outputs=actions)
      return

    self.model = Model(inputs=states, outputs=actions)

    action_gradients = Input(shape=(self.action_size,))
    loss = K.mean(-action_gradients * actions)

    optimizer = Adam(lr=0.0001)
    updates_op = optimizer.get_updates(
        params=self.model.trainable_weights, loss=loss)
    self.train_fn = K.function(
        inputs=[self.model.input, action_gradients, K.learning_phase()],
        outputs=[],
        updates=updates_op)

  def target_predict(self, states):
    return self.target.predict_on_batch(states)

  def soft_update(self, tau):
    weights = np.array(self.model.get_weights())
    target_weights = np.array(self.target.get_weights())

    next_weights = tau * weights + (1 - tau) * target_weights
    self.target.set_weights(next_weights)

  def predict(self, state):
    return self.model.predict(state)


class Critic:
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size

    self.build_model()
    self.build_model(True)
    self.target.set_weights(self.model.get_weights())

  def build_model(self, target=False):
    states = Input(shape=(self.state_size,), name='states')
    actions = Input(shape=(self.action_size,), name='actions')
    net = Dense(units=20, activation='relu')(states)
    net = Add()([net, actions])
    net = Dense(units=20, activation='relu')(net)
    Q_values = Dense(units=1,name='Q_values')(net)

    if target:
      self.target = Model(inputs=[states, actions], outputs=Q_values)
      return

    self.model = Model(inputs=[states, actions], outputs=Q_values)

    optimizer = Adam(lr=0.05)
    self.model.compile(optimizer=optimizer, loss='mse')

    action_gradients = K.gradients(Q_values, actions)

    self.get_action_gradients = K.function(
        inputs=[*self.model.input, K.learning_phase()],
        outputs=action_gradients)

  def target_predict(self, states, actions):
    return self.target.predict_on_batch([states, actions])

  def train_on_batch(self, states, actions, Q_targets):
    self.model.train_on_batch(x=[states, actions], y=Q_targets)

  def soft_update(self, tau):
    weights = np.array(self.model.get_weights())
    target_weights = np.array(self.target.get_weights())

    next_weights = tau * weights + (1 - tau) * target_weights
    self.target.set_weights(next_weights)

  def target_predict_on_batch(self, states, actions):
    return self.target.predict_on_batch([states, actions])


class ReplayBuffer:
  def __init__(self, buf_size):
    self.buffer = deque(maxlen=buf_size)
    self.experience = namedtuple("Experience",
                                 field_names=["s0", "a", "r", "s1", "done"])

  def add(self, state, action, reward, next_state, done):
    e = self.experience(state, action, reward, next_state, done)
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


class DDPG():
  def __init__(self):
    self.state_size = 2
    self.action_size = 1

    self.actor = Actor(self.state_size, self.action_size)
    self.critic = Critic(self.state_size, self.action_size)

    self.exploration_mu = 0
    self.exploration_theta = 0.05
    self.exploration_sigma = 0.25
    self.noise = OUNoise(self.action_size, self.exploration_mu,
                         self.exploration_theta, self.exploration_sigma)

    self.buffer_size = 10000
    self.batch_size = 128
    self.memory = ReplayBuffer(self.buffer_size)

    self.gamma = 0.999
    self.tau_actor = 0.1
    self.tau_critic = 0.2

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
    state = np.reshape(state, [-1, self.state_size])
    true_action = self.actor.predict(state)[0]
    noise = self.noise.sample()
    action = np.clip(true_action*.2 + noise, -1, 1)
    return list(action), true_action

  def learn(self, experiences):
    def to_list(array, key):
      return [getattr(x, key) for x in array if x is not None]

    states = np.vstack(to_list(experiences, 's0'))
    actions = np.array(to_list(experiences, 'a')).astype(np.float32).reshape(-1, self.action_size)
    rewards = np.array(to_list(experiences, 'r')).astype(np.float32).reshape(-1, 1)
    next_states = np.vstack(to_list(experiences, 's1'))

    actions_next = self.actor.target_predict(next_states)
    Q_targets_next = self.critic.target_predict_on_batch(next_states, actions_next)

    Q_targets = rewards + self.gamma * Q_targets_next
    self.critic.train_on_batch(states, actions, Q_targets)

    action_gradients = np.reshape(self.critic.get_action_gradients([states, actions, 0]), (-1, self.action_size))
    self.actor.train_fn([states, action_gradients, 1])

    self.critic.soft_update(self.tau_critic)
    self.actor.soft_update(self.tau_actor)

class MountainCarContinuous():
  def __init__(self, render=False):
    self.env = self.new_env()
    self.agent = DDPG()
    self.epoch = namedtuple("epoch",
                   field_names=["sum_reward", "done", "action_mean", "action_stddev", "steps"])

  def new_env(self):
    return gym.make('MountainCarContinuous-v0').unwrapped

  def preprocess_state(self, state):
    s = np.array(state)
    s[0] = ((state[0] + 1.2) / 1.8) * 2 - 1
    s[1] = ((state[1] + 0.07) / 0.14) * 2 - 1
    return s

  def run_epoch(self, max_steps, render=False, training=True):
    state = self.preprocess_state(self.env.reset())
    self.agent.reset_episode(state)
    actions_list = []
    sum_reward = 0
    steps = 0
    while steps < max_steps:
      steps += 1
      noisy_action, true_action = self.agent.act(state)
      action = noisy_action if training else true_action

      next_state, reward, done, info = self.env.step(action)
      next_state = self.preprocess_state(next_state)
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
                max_steps=1000, verbose=1, render=False):
    solved = False
    train_hist = []
    test_hist = []

    for epoch in range(1, max_epochs+1):
      train_epoch = self.run_epoch(max_steps=max_steps)
      test_epoch = self.run_epoch(max_steps=max_steps, training=False, render=render)

      train_hist.append([train_epoch.sum_reward, train_epoch.steps])
      test_hist.append([test_epoch.sum_reward, test_epoch.steps])

      if epoch > n_solved:
        train_running = np.mean([r for r, s in train_hist][-n_solved:])
        test_running = np.mean([r for r, s in test_hist][-n_solved:])
      else:
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
    o = MountainCarContinuous()
    o.run_model(max_epochs=50, n_solved=10)
    o.run_model(max_epochs=5, n_solved = 1, verbose=0, render=True)
