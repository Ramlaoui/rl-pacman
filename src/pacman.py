# Adapted from https://github.com/hbokmann/Pacman/blob/master/pacman.py
import pygame

black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
blue = (0, 0, 255)
green = (0, 255, 0)
yellow = (255, 255, 0)
purple = (255, 0, 255)


class Wall(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height, color, gui=True):
        pygame.sprite.Sprite.__init__(self)

        if gui:
            self.image = pygame.Surface([width, height])
            self.image.fill(color)
            self.rect = self.image.get_rect()
        else:
            self.rect = pygame.Rect(x, y, width, height)

        self.rect.top = y
        self.rect.left = x


class Block(pygame.sprite.Sprite):
    def __init__(self, color, width, height, gui=True):
        pygame.sprite.Sprite.__init__(self)

        if gui:
            self.image = pygame.Surface([width, height])
            self.image.fill(white)
            self.image.set_colorkey(white)
            pygame.draw.ellipse(self.image, color, [0, 0, width, height])
            self.rect = self.image.get_rect()
        else:
            self.rect = pygame.Rect(0, 0, width, height)


class Player(pygame.sprite.Sprite):

    change_x = 0
    change_y = 0

    def __init__(self, x, y, filename, gui=True):
        pygame.sprite.Sprite.__init__(self)

        if gui:
            self.image = pygame.image.load(filename).convert()
            self.rect = self.image.get_rect()
        else:
            self.rect = pygame.Rect(x, y, 32, 32)
        self.rect.top = y
        self.rect.left = x
        self.prev_x = x
        self.prev_y = y

    def prevdirection(self):
        self.prev_x = self.change_x
        self.prev_y = self.change_y

    def changespeed(self, x, y):
        self.change_x += x
        self.change_y += y

    def update(self, walls, gate):

        old_x = self.rect.left
        new_x = old_x + self.change_x
        prev_x = old_x + self.prev_x
        self.rect.left = new_x

        old_y = self.rect.top
        new_y = old_y + self.change_y
        prev_y = old_y + self.prev_y

        x_collide = pygame.sprite.spritecollide(self, walls, False)
        if x_collide:
            self.rect.left = old_x
        else:
            self.rect.top = new_y

            y_collide = pygame.sprite.spritecollide(self, walls, False)
            if y_collide:
                self.rect.top = old_y

        if gate != False:
            gate_hit = pygame.sprite.spritecollide(self, gate, False)
            if gate_hit:
                self.rect.left = old_x
                self.rect.top = old_y


class Ghost(Player):
    def changespeed(self, list, ghost, turn, steps, l):
        try:
            z = list[turn][2]
            if steps < z:
                self.change_x = list[turn][0]
                self.change_y = list[turn][1]
                steps += 1
            else:
                if turn < l:
                    turn += 1
                elif ghost == "clyde":
                    turn = 2
                else:
                    turn = 0
                self.change_x = list[turn][0]
                self.change_y = list[turn][1]
                steps = 0
            return [turn, steps]
        except IndexError:
            return [0, 0]
