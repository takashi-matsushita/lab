import collections

import gym

import numpy as np
import scipy.signal

import keras
from keras import layers
from keras.models import Model
from keras import backend as K
from keras import optimizers


class History:
  def __init__(self):
    self.n_sample = 0
    self.n_epoch = 0
    self.mean_rtg = 0.
    self.var_rtg = 0.

  def reset(self):
    self.history = {
      'states': [],
      'actions': [],
      'reward': [],
      }

  def append(self, state, action, reward):
    self.history['states'].append(state)
    self.history['actions'].append(action)
    self.history['reward'].append(reward)
    self.n_sample += 1

  def get_state(self):
    rc = np.array(self.history['states'])
    return rc

  def get_action(self):
    return np.array(self.history['actions'])

  def get_reward(self):
    return np.array(self.history['reward'])

  def get_reward_to_go(self, discount_rate=.99):
    R = self.get_reward()
    rtg = np.zeros_like(R, dtype=np.float32)
    running_sum = 0.

    rtg = scipy.signal.lfilter([1], [1, float(-discount_rate)], R[::-1], axis=0)[::-1]
    mean = np.mean(rtg)
    variance = np.var(rtg)

    size = len(rtg)
    if self.n_epoch:
      self.mean_rtg = (self.n_sample*self.mean_rtg + size*mean)/(self.n_sample + size)
      self.var_rtg = ((self.n_sample-1)*self.var_rtg + (size-1)*variance)/(self.n_sample + size)
    else:
      self.mean_rtg = mean
      self.var_rtg = variance

    self.n_epoch += 1
    rtg = (rtg - self.mean_rtg)/(5*np.sqrt(self.var_rtg))

    return rtg


""" Reward-to-Go Policy Gradient
    ref: https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
"""
class Agent(object):

  def policy_optimization_loss(action_onehot, rtg):
    def loss(y_true, y_pred):
      ## loss function
      #  Sum[ log(pi[a|s]) ] * R[tau]
      log_action_prob = K.sum(K.log(y_pred) * action_onehot, axis=1)
      return -K.mean(log_action_prob*rtg)
    return loss

  def __init__(self, nstate, naction):
    self.nstate = nstate
    self.naction = naction

    self.build_model()

  def build_model(self):
    ## inputs
    states = layers.Input(shape=(self.nstate,))
    action_onehot = layers.Input(shape=(self.naction,), name="action_onehot")
    rtg = layers.Input(shape=(1,), name="reward_to_go")

    ## policy network
    net = layers.Dense(16)(states)
    net = layers.Activation("relu")(net)
    net = layers.Dense(16)(net)
    net = layers.Activation("relu")(net)
    net = layers.Dense(self.naction)(net)
    actions = layers.Activation("softmax")(net)

    self.model = Model(inputs=[states, action_onehot, rtg], outputs=actions)
    adam = optimizers.Adam()
    self.model.compile(optimizer=adam, loss=[Agent.policy_optimization_loss(action_onehot, rtg)])

  def get_action(self, state, training=True):
    if len(state.shape) == 1:
      state = np.expand_dims(state, axis=0)

    dummy_onehot = np.zeros(shape=(len(state), self.naction))
    dummy_rtg = np.zeros(shape=(len(state),))
    action_prob = np.squeeze(self.model.predict([state, dummy_onehot, dummy_rtg]))
    rc = None
    if training:
      rc = np.random.choice(np.arange(self.naction), p=action_prob)
    else:
      rc = np.argmax(action_prob)

    return rc

  def fit(self, history, batch_size=32):
    action_onehot = keras.utils.to_categorical(history.get_action(), num_classes=self.naction)
    rtg = history.get_reward_to_go(discount_rate=.99)
    self.model.fit([history.get_state(), action_onehot, rtg], [history.get_action()], batch_size=batch_size, verbose=0)


def run_episode(env, agent, history, render=False, training=True):
  done = False
  batch_size = 128
  total_reward = 0

  s0 = env.reset()
  history.reset()

  while not done:
    a = agent.get_action(s0, training)
    s1, r, done, info = env.step(a)
    if render: env.render()
    total_reward += r
    history.append(s0, a, r)
    s0 = s1
    run_episode.counter += 1

    if done and training:
      agent.fit(history, batch_size)

  return total_reward


if __name__ == '__main__':
  model = "CartPole-v0"
  maxlen = 100
  min_average = 195

  env = gym.make(model)
  nstate  = env.env.observation_space.shape[0]
  naction = env.env.action_space.n
  agent = Agent(nstate, naction)

  history = History()
  run_episode.counter = 0
  rb = collections.deque(maxlen=100)
  counter = 0
  while True:
    render = (counter % maxlen == 0)
    reward = run_episode(env, agent, history, render)
    rb.append(reward)
    counter += 1
    average = np.mean(rb)
    if counter % (maxlen/10) == 0:
      print('epoch = {:4d} running mean = {:6.2f}'.format(counter, average))

    if counter > maxlen-1 and average > min_average:
      print('done ', counter, average)
      break

  reward = run_episode(env, agent, history, render=True, training=False)
  print(reward)

# eof
