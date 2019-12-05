#!/usr/bin/env python3
import gym
import gym_flappy_fish
from collections import namedtuple
import numpy as np


from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras import losses
from keras.utils import np_utils

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70

def create_model(obs_size, hidden_size, n_actions):
    model = Sequential()
    model.add(Dense(128, input_dim = obs_size))
    model.add(Activation('relu'))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(n_actions, activation='softmax'))
    print(model.summary())
    return model

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

def iterate_batches(env, model, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()

    while True:
        obs_v = np.array(obs).reshape(1,len(obs))
        #print(f'obs: {obs_v}, len obs: {len(obs)} and obs_v {obs_v}')
        act_probs = model.predict(obs_v)[0]
        print(f'obs: {obs_v}, action = {act_probs}')
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, fish_pipes = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))
        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))
    return train_obs, train_act, reward_bound, reward_mean


if __name__ == "__main__":
    env = gym.make("flappy_fish-v0")
    obs_size = env.observation_space.shape[0]
    print(f"obs_size: {obs_size}")
    n_actions = env.action_space.n
    print(n_actions)

    model = create_model(obs_size, HIDDEN_SIZE, n_actions)
    model.compile(loss = losses.categorical_crossentropy, optimizer = Adam(lr=0.01))

    for iter_no, batch in enumerate(iterate_batches(env, model, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        obs = np.array(obs_v).reshape((-1,obs_size))
        #print(obs)
        #print(f'acts_v: {acts_v}')
        acts = np_utils.to_categorical(acts_v, num_classes=2)
        history = model.fit(obs, acts)

        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
            iter_no, history.history['loss'][0], reward_m, reward_b))
        show = False
        if(reward_m > 100):
            show = True
        if show:
            if iter_no % 10 == 1:
                obs = env.reset()
                total_reward = 0
                obs = env.reset()
                while True:
                    obs_v = np.array(obs).reshape(1,len(obs))
                    #print(f'obs: {obs_v}, len obs: {len(obs)} and obs_v {obs_v}')
                    act_probs = model.predict(obs_v)[0]
                    action = np.random.choice(len(act_probs), p=act_probs)
                    next_obs, reward, is_done, fish_pipes = env.step(action)
                    total_reward += reward
                    
                    if done:
                        print(f'End. Total reward: {total_reward}')
                        break
                    env.render()
            

        if reward_m > 199:
            print("Solved!")
            break