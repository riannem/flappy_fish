import os
import gym
import pygame
from gym import error, spaces, utils
from gym.utils import seeding
from .fish import Fish
from .pipe_list import PipeList
from .config import *
import numpy as np

class FlappyFishEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.fish = Fish(pygame.image.load(os.path.join("gym_flappy_fish", "envs", "images", "fish.png")))
        self.pipes = PipeList()
        self.counter = 0

        self.action_space = spaces.Discrete(2)

        #obs: fish.velocity, fish.y, x-position pipes, bottom of gap, top of gap
        min_obs = np.array([-50,0,0,0,0], dtype=np.float32)
        max_obs = np.array([50,600,400,600,600], dtype=np.float32)
        self.observation_space = spaces.Box(min_obs, max_obs, dtype = np.float32)

        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Flappy Fish')
        self.bg = pygame.image.load(os.path.join("gym_flappy_fish","envs", "images", "underwater.png"))
    
        self.font = pygame.font.Font("freesansbold.ttf", 20)

        self.state = None

    def step(self, action):
        reward = 0
        self.fish.update(action)
        self.pipes.update()

        done = True
        collision = self.pipes.check_collision(self.fish)
        off_screen = self.fish.check_off_screen()

        if not collision and not off_screen:
            reward = 1
            done = False
            self.counter += 1
            if self.fish.check_through_pipe(self.pipes.l[0]):
                reward = 10
        else:
            reward = -1000

        self.state = (self.fish.velocity, self.fish.y, self.pipes.l[0].x, self.pipes.l[0].y_bottom, self.pipes.l[0].y_top)
        return np.array(self.state), reward, done, (self.fish, self.pipes)

    def reset(self):
        self.fish.reset_fish()
        self.pipes = PipeList()
        self.counter = 0
        self.state = (self.fish.velocity, self.fish.y, self.pipes.l[0].x, self.pipes.l[0].y_bottom, self.pipes.l[0].y_top)
        return np.array(self.state)
        
        
    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()

        self.screen.blit(self.bg, (0,0))
        self.fish.draw(self.screen)
        self.pipes.draw(self.screen)
        text = self.font.render("Fish alive for {} screens".format(self.counter),
                        True, (250, 250, 200), None)
        text_rect = text.get_rect()
        text_rect.center = (WIDTH // 2, 30)
        self.screen.blit(text, text_rect)
        pygame.display.flip()
        