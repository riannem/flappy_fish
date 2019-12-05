import pygame
import gc
import numpy as np
from keras.models import Sequential, clone_model
from keras.layers import Dense, Activation
from keras.optimizers import Adam

from .config import *


class Fish():
    """
    Fish class, predict action, update position and
    draw the bird.
    """

    def __init__(self, image):

        self.image = image
        self.x = 2 * CIRCLE_RADIUS
        self.y = int(HEIGHT/2.0)
        self.radius = CIRCLE_RADIUS
        self.gravity_force = GRAVITY / BIRD_MASS
        self.velocity = 0
        self.acceleration = 0



    def reset_fish(self):
        """Reset Fish to start posistion"""
        self.y = int(HEIGHT/2.0)
        self.velocity = 0
        self.score = 0


    def draw(self, screen):
        """
        Draw the Fish object to the screen

        :param screen: pygame screen object
        """
        r = pygame.Rect(self.x-self.radius,
                        self.y-self.radius,
                        2*self.radius,
                        2*self.radius)
        screen.blit(self.image, r)


    def check_off_screen(self):
        """
        Check if Fish position is off screen

        :retrun: boolean, True when off screen
        """
        if self.y - self.radius <= 0 or self.y + self.radius >= HEIGHT:
            return True

    def check_through_pipe(self, pipe):
        """
        Check if Fish passed the pipe

        :return: boolean, True when through the pipe
        """
        if (self.x >= pipe.x+PIPE_WIDTH - 4) and (self.x < pipe.x +PIPE_WIDTH):
            return True

    def update(self, action):
        """
        predict (using the Fish neural network) an action based on environment
        input, then update acceleration, velocity and finally, bird position.

        :param pipe_info: tuple with environment variables
        """

        if (action == 1) and (self.velocity >= 0.0):
            self.acceleration += JUMP_FORCE
        else:
            self.acceleration += self.gravity_force
        # update position
        self.velocity += self.acceleration
        self.y += int(self.velocity)
        self.acceleration = 0
