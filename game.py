import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import sys

pygame.init()
font = pygame.font.SysFont('Arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 10000

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Yılan AI')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w//2, self.h//2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2*BLOCK_SIZE, self.head.y)
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            if self.food not in self.snake:
                break

    def play_step(self, action):
        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        action = np.array(action)  # Güvenlik için

        old_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0

        # Çarpışma kontrolü veya çok uzun süre aynı yerde dönme
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            reward = -10
            return reward, True, self.score

        # Yem yedi mi?
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
            self.frame_iteration = 0  # yem yerse sayaç sıfırlansın
        else:
            self.snake.pop()

            # Her adım için küçük ceza (döngü engeli)
            reward = -0.01

            # Uzaklaşma/yaklaşma cezası/ödülü
            new_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
            if new_distance < old_distance:
                reward += 0.1
            elif new_distance > old_distance:
                reward -= 0.1

        self._update_ui()
        self.clock.tick(SPEED)

        return reward, False, self.score

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        if pt.x < 0 or pt.x >= self.w or pt.y < 0 or pt.y >= self.h:
            return True

        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Skor: " + str(self.score), True, WHITE)
        self.display.blit(text, [10, 10])
        pygame.display.flip()

    def _move(self, action):
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clockwise[idx]  # düz
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clockwise[(idx + 1) % 4]  # sağ
        else:  # [0, 0, 1]
            new_dir = clockwise[(idx - 1) % 4]  # sol

        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
