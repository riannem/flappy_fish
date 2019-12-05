from gym.envs.registration import register

register(
    id = 'flappy_fish-v0',
    entry_point = 'gym_flappy_fish.envs:FlappyFishEnv'
)