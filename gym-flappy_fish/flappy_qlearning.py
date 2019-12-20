import gym
import gym_flappy_fish
import numpy as np
import random
import datetime
import gc
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque

class DQN:
    def __init__(self, env):
        self.env     = env
        self.memory  = deque(maxlen=10000)

        self.gamma = 0.95
        self.epsilon = 0.95
        self.epsilon_min = 0.00001
        self.epsilon_decay = 0.999
        self.learning_rate = 0.005
        self.tau = 0.5

        self.model        = self.create_model()
        self.target_model = self.create_model()
        self.max_reward = 0
        self.best_memory = []
        self.episode_memory = []

    def create_model(self):
        model   = Sequential()
        state_shape  = self.env.observation_space.shape
        model.add(Dense(64, input_dim=state_shape[0], activation="relu"))
        #model.add(Dense(16, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        prediction = self.model.predict(state.reshape(-1, len(state)))[0]
        return np.argmax(prediction)

    def remember(self, state, action, reward, new_state, done):
        self.episode_memory.append([state, action, reward, new_state, done])

    def clear_memory(self):
        self.episode_memory = []
        gc.collect()

    def replay(self, tot_reward):
        for s in self.episode_memory: self.memory.append(s)
        batch_size = 500
        if batch_size > len(self.memory):
            batch_size = len(self.memory)
        batch = random.sample(list(self.memory), batch_size)

        for sample in batch + self.best_memory:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state.reshape(-1, len(state)))
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state.reshape(-1,len(new_state)))[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state.reshape(-1, len(state)), target, epochs=1, verbose=0)
        if tot_reward > self.max_reward:
            self.max_reward = tot_reward
            self.best_memory = self.episode_memory

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)

def main():
    env     = gym.make("flappy_fish-v0")

    trials  = 50000
    trial_len = 10000

    dqn_agent = DQN(env=env)
    steps = []
    for trial in range(trials):
        cur_state = env.reset()
        tot_reward = 0
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)
            if step % 10 == 0:
                env.render()
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            tot_reward += reward

            cur_state = new_state
            if done:
                dqn_agent.replay(tot_reward)
                dqn_agent.target_train()
                dqn_agent.clear_memory()
                break
        if tot_reward < 20000:
            print(f"Failed to complete in trial {trial}, reward: {tot_reward} in {step} steps")
            if tot_reward > 2000:
                dqn_agent.save_model("trial-{}.model".format(trial))
                print(datetime.datetime.now())
        else:
            print("Completed in {} trials".format(trial))
            dqn_agent.save_model("success.model")
            break

if __name__ == "__main__":
    print(datetime.datetime.now())
    main()