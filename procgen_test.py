import gym
import random
import numpy as np


print("procgen:procgen-locacoinrun-v0")
env = gym.make("procgen:procgen-locacoinrun-v0", locacoinrun_draw_bars=True)
env.reset()

episode_reward = 0
env.seed()
while True:
  action = env.action_space.sample()
  obs, reward, done, _ = env.step(action)
  #print(obs.shape)
  episode_reward += reward
  if done:
    #env.step(action)
    print('Reward: %s' % episode_reward)
    break