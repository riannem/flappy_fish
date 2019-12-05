import random
import pygame

from .config import *




class Pipe():

    def __init__(self):
        self.x = WIDTH
        self.y = random.randint(HALF_GAP_SIZE + PIPE_MARGIN,
                                HEIGHT - PIPE_MARGIN - HALF_GAP_SIZE)
        self.y_bottom = self.y + HALF_GAP_SIZE
        self.y_top = self.y - HALF_GAP_SIZE

    def update(self):
        self.x -= PIPE_SPEED

    def draw(self, screen):

        # Get Rect boundaries for top and bottom pipes
        r1 = pygame.Rect(self.x, 0,
            PIPE_WIDTH, self.y_top)
        r2 = pygame.Rect(self.x, self.y_bottom,
                         PIPE_WIDTH, HEIGHT - self.y_top)
        # Draw pipes
        pygame.draw.rect(screen, (50, 150, 50), r1)
        pygame.draw.rect(screen, (50, 150, 50), r2)
