import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import gym

import matplotlib.pyplot as plt
plt.style.use('bmh')
import matplotlib
matplotlib.rcParams['font.family'] = 'IPAPGothic'


def reset_seeds():
  random.seed(9949)
  np.random.seed(9967)
  import tensorflow as tf; tf.set_random_seed(9973)


def policy_table(env, nstate, naction, nplay=100):
  ## initialise policy table
  policy = np.zeros((nstate, naction))

  for _ in range(nplay):
    s0 = env.reset()
    done = False

    while not done:
      ## decide action to take
      if np.sum(policy[s0, :]) == 0:
        action = np.random.randint(0, naction)
      else:
        action = np.argmax(policy[s0, :])

      ## update policy table
      s1, reward, done, info = env.step(action)
      policy[s0, action] += reward
      s0 = s1

  return policy


def policy_table_qlearning(env, nstate, naction, nplay=100):
  ## initialise policy table
  policy = np.zeros((nstate, naction))

  y = 0.95  # discount factor
  lr = 0.8  # learning rate

  for _ in range(nplay):
    s0 = env.reset()
    done = False

    while not done:
      ## decide action to take
      if np.sum(policy[s0, :]) == 0:
        action = np.random.randint(0, naction)
      else:
        action = np.argmax(policy[s0, :])

      ## update policy table, considering future reward
      s1, reward, done, info = env.step(action)
      policy[s0, action] += reward + lr*(y*np.max(policy[s1, :]) - policy[s0, action])
              
      s0 = s1

  return policy


def policy_table_egreedy_qlearning(env, nstate, naction, nplay=100):
  ## initialise policy table
  policy = np.zeros((nstate, naction))

  y = 0.95      # discount factor
  lr = 0.8      # learning rate
  eps = 0.5     # epsilon
  decay = 0.999 # decay factor

  for _ in range(nplay):
    s0 = env.reset()
    done = False
    eps *= decay

    while not done:
      ## decide action to take
      if (np.random.random() < eps) or (np.sum(policy[s0, :]) == 0):
        action = np.random.randint(0, naction)
      else:
        action = np.argmax(policy[s0, :])

      ## update policy table, considering future reward
      s1, reward, done, info = env.step(action)
      policy[s0, action] += reward + lr*(y*np.max(policy[s1, :]) - policy[s0, action])
              
      s0 = s1

  return policy


env = gym.make('NChain-v0')
nstate = env.observation_space.n
naction = env.action_space.n

reset_seeds()
policy_0 = policy_table(env, nstate, naction)
policy_1 = policy_table_qlearning(env, nstate, naction)
policy_2 = policy_table_egreedy_qlearning(env, nstate, naction)


def run_game(policy, env):
  s0 = env.reset()
  sum_reward = 0
  done = False
  while not done:
    action = np.argmax(policy[s0, :])
    s1, reward, done, info = env.step(action)
    sum_reward += reward
  return sum_reward


def compare_algorithms(env, nstate, naction, nplay=100):
  winner = np.zeros((3,))

  for _ in range(nplay):
    print('inf> loop = {}'.format(_))
    policy_rl = policy_table(env, nstate, naction)
    policy_qrl = policy_table_qlearning(env, nstate, naction)
    policy_egqrl = policy_table_egreedy_qlearning(env, nstate, naction)
    rl = run_game(policy_rl, env)
    qrl = run_game(policy_qrl, env)
    egqrl = run_game(policy_egqrl, env)
    w = np.argmax(np.array([rl, qrl, egqrl]))
    print('inf> winner = {}'.format(w))
    winner[w] += 1
  return winner


compare_algorithms(env, nstate, naction)


def deep_q_learning(env, nstate, naction, nplay=100):
  ## build deep Q-network
  from keras.models import Sequential
  from keras.layers import InputLayer, Dense
  from keras.callbacks import CSVLogger

  path_log = 'rl_gym-log.csv'
  callbacks = [
    CSVLogger(filename=path_log, append=True),
    ]
  if os.path.isfile(path_log): os.remove(path_log)

  model = Sequential()
  model.add(InputLayer(batch_input_shape=(1, nstate)))
  model.add(Dense(10, activation='sigmoid'))
  model.add(Dense(naction, activation='linear'))
  model.compile(loss='mse', optimizer='adam', metrics=['mae'])


  y = 0.95      # discount factor
  eps = 0.5     # epsilon
  decay = 0.999 # decay factor

  sum_rewards = []
  for ii in range(nplay):
    s0 = env.reset()
    eps *= decay
    done = False
    if ii % 10 == 0:
      print("loop {} of {}".format(ii+1, nplay))

    sum_reward = 0.
    while not done:
      ## decide action to take
      if (np.random.random() < eps):
        action = np.random.randint(0, naction)
      else:
        action = np.argmax(model.predict(np.identity(nstate)[s0:s0+1]))

      s1, reward, done, info = env.step(action)

      ## update deep Q-network
      target = reward + y * np.max(model.predict(np.identity(nstate)[s1:s1+1]))
      target_vec = model.predict(np.identity(nstate)[s0:s0 + 1])[0]
      target_vec[action] = target 
      model.fit(np.identity(nstate)[s0:s0+1], target_vec.reshape(-1, naction),
                callbacks=callbacks, epochs=1, verbose=0)
      s0 = s1
      sum_reward += reward
    sum_rewards.append(sum_reward)

  # construct policy table
  policy = np.zeros((nstate, naction))
  for ii in range(nstate):
    policy[ii] = model.predict(np.identity(nstate)[ii:ii+1])

  return policy, sum_rewards


reset_seeds()
policy_3, sum_rewards = deep_q_learning(env, nstate, naction, nplay=1000)


def print_policy(policy):
  for ii in range(policy.shape[0]):
    print('state={} action={}'.format(ii, np.argmax(policy[ii, :])))

print('simple learning')
print_policy(policy_0)
print('q-learning')
print_policy(policy_1)
print('epsilon greedy q-learning')
print_policy(policy_2)
print('deep q-learning')
print_policy(policy_3)

### plot learning curve
df = pd.DataFrame({'rewards':sum_rewards})  
df.plot()
plt.xlabel('# of play')
plt.ylabel('rewards')
plt.show()

plt.plot(df['rewards'].rolling(10).mean())

# eof
