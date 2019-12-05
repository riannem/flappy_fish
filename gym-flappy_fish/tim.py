import os
import os
import gym
import gym_flappy_fish
import pygame
import time
import numpy as np
import random
from gym_flappy_fish.envs.config import *

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque

def create_model():

    model = Sequential()
    model.add(Dense(25, input_dim=5, activation="relu"))
    model.add(Dense(25, activation="relu"))
    model.add(Dense(2, activation="relu"))
    model.compile(loss="mean_squared_error",
        optimizer=Adam(lr=0.002))
    return model

class Memory():
    def __init__(self):
        self.memory = deque(maxlen=2000)

    def add(self, obs, new_obs, reward, done, action):
        self.memory.append([obs, new_obs, reward, done, action])

    def sample(self, size=200):
        if len(self.memory) >= size:
            return random.sample(self.memory, size)
        else:
            return list(self.memory)

def update_model(model, mem):
    gamma = 0.9
    sample = mem.sample()
    obs = [i[0] for i in sample]
    action = [i[-1] for i in sample]
    rewards = [i[2] for i in sample]
    for i, ob in enumerate(obs):
        ac = model.predict(ob.reshape(-1,5))
        print(ac)
        temp = np.argmax(ac)
        ac[0][temp] = 1
        ac = np.array([[1,0]])
        # print(ac)
        model.fit(ob.reshape(-1,5), ac[0].reshape(-1,2), epochs=10, verbose=0)
    return model

def act(model, ep, env):
    action = np.argmax(model.predict(obs.reshape(-1,5)))
    return action


if __name__ == '__main__':

    #create env and get observations
    env = gym.make('flappy_fish-v0')
    counter = 0
    obs = env.reset()
    
    model = create_model()
    mem = Memory()

    epsilon = 1.0


    for ep in range(100):
        obs = env.reset()
        cum_reward = 0
        for step in range(100000):
            action = act(model, ep, env)
            new_obs, reward, done, _ = env.step(action)
            cum_reward += reward
            env.render()
            if done:
                print(f"Episode: {ep}, Steps: {step}, tot_reward: {cum_reward}")
                mem.add(obs, new_obs, -1, done, action)
                model = update_model(model, mem)
                break
            mem.add(obs, new_obs, reward, done, action)

            obs = new_obs

