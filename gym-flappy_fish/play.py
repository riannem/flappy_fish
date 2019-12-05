import os
import gym
import gym_flappy_fish
import pygame
import time
import numpy as np
from gym_flappy_fish.envs.config import *

if __name__ == '__main__':
    
    #create env and get observations
    env = gym.make('flappy_fish-v0')
    counter = 0
    obs = env.reset()

    counter = 0
    total_reward = 0

    while True:
        action = np.random.choice(2, 1, p=[0.9, 0.1])
        obs, reward, done, _ = env.step(action)
        counter += 1
        total_reward += reward
        if done:
            print(f'reward: {total_reward} in {counter} steps')
            break
        env.render()
