# Adapted from https://github.com/hbokmann/Pacman/blob/master/pacman.py
import pygame
import numpy as np
from src.pacman import Wall, Block, Player, Ghost

black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
blue = (0, 0, 255)
green = (0, 255, 0)
yellow = (255, 255, 0)
purple = (255, 0, 255)

pygame.mixer.init()


class Monster:
    def __init__(self, x, y, name, directions):
        self.x = x
        self.y = y
        self.name = name
        self.directions = directions
        self.l = len(directions) - 1
        self.turn = 0
        self.steps = 0

    def reset(self):
        self.turn = 0
        self.steps = 0


class PlayerPos:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Gate:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


def setupRoomOne(walls, all_sprites_list, gui=True):
    wall_list = pygame.sprite.RenderPlain()

    # Loop through the list. Create the wall, add it to the list
    for item in walls:
        wall = Wall(item[0], item[1], item[2], item[3], blue, gui=gui)
        wall_list.add(wall)
        all_sprites_list.add(wall)

    # return our new list
    return wall_list


def setupGate(gate_, all_sprites_list, gui=True):
    gate = pygame.sprite.RenderPlain()
    gate.add(Wall(gate_.x, gate_.y, gate_.width, gate_.height, white, gui=gui))
    all_sprites_list.add(gate)
    return gate


class Event:
    def __init__(self, action):
        self.key = action


class Game:
    def __init__(self, walls, gate, player, monsters, gui=False, ai=False):
        self.walls = walls
        self.gate_init = gate
        self.player = player
        self.monsters = monsters
        self.gui = gui
        self.ai = ai
        self.width = 606
        self.height = 606
        self.ACTIONS_MAP = {
            0: pygame.K_LEFT,  # "left"
            1: pygame.K_RIGHT,  # "right"
            2: pygame.K_UP,  # "up"
            3: pygame.K_DOWN,  # "down"
            4: None,
        }

    def reset(self):
        pygame.init()
        if self.gui:
            self.screen = pygame.display.set_mode([self.width, self.height])
            pygame.display.set_caption("Pacman")
            self.background = pygame.Surface(self.screen.get_size())
            self.background = self.background.convert()
            self.background.fill(black)
            self.clock = pygame.time.Clock()
            pygame.font.init()
            self.font = pygame.font.Font("freesansbold.ttf", 24)
        self.all_sprites_list = pygame.sprite.RenderPlain()
        self.block_list = pygame.sprite.RenderPlain()
        self.monsta_list = pygame.sprite.RenderPlain()
        self.pacman_collide = pygame.sprite.RenderPlain()
        self.wall_list = setupRoomOne(self.walls, self.all_sprites_list, gui=self.gui)
        self.gate = setupGate(self.gate_init, self.all_sprites_list, gui=self.gui)
        self.p_turn = 0
        self.p_steps = 0
        self.b_turn = 0
        self.b_steps = 0
        self.i_turn = 0
        self.i_steps = 0
        self.c_turn = 0
        self.c_steps = 0
        self.Pacman = Player(
            self.player.x, self.player.y, "images/Trollman.png", gui=self.gui
        )
        self.all_sprites_list.add(self.Pacman)
        self.pacman_collide.add(self.Pacman)
        self.Monsters = []
        for monster in self.monsters:
            monster.reset()
            self.Monsters.append(
                Ghost(monster.x, monster.y, f"images/{monster.name}.png", gui=self.gui)
            )
            self.monsta_list.add(self.Monsters[-1])
            self.all_sprites_list.add(self.Monsters[-1])
        self.i_blocks = 0
        self.blocks_x_y_to_index = {}
        for row in range(19):
            for column in range(19):
                if (row == 7 or row == 8) and (
                    column == 8 or column == 9 or column == 10
                ):
                    continue
                else:
                    block = Block(yellow, 4, 4)

                    block.rect.x = (30 * column + 6) + 26
                    block.rect.y = (30 * row + 6) + 26

                    b_collide = pygame.sprite.spritecollide(
                        block, self.wall_list, False
                    )
                    p_collide = pygame.sprite.spritecollide(
                        block, self.pacman_collide, False
                    )
                    if b_collide:
                        continue
                    elif p_collide:
                        continue
                    else:
                        self.blocks_x_y_to_index[(block.rect.x, block.rect.y)] = (
                            self.i_blocks
                        )
                        self.i_blocks += 1
                        self.block_list.add(block)
                        self.all_sprites_list.add(block)
        self.bll = len(self.block_list)
        self.score = 0
        self.prev_score = None
        self.done = False
        self.i = 0
        self.reward = 0
        self.obs = self.get_observation()
        return self.obs

    def get_observation(self, prev_obs=None):
        """
        Observation vector:
        - 0: Pacman x
        - 1: Pacman y
        - 2: Pacman change_x
        - 3: Pacman change_y
        - i + 4: Monster i x
        - i + 5: Monster i y
        - i + 6: Monster i change_x
        - i + 7: Monster i change_y
        - 4 + len(monsters) + j: if 1, block j is present, if 0 it is not
        """
        if prev_obs is not None:
            prev_obs = np.array(prev_obs)
        obs = [
            self.Pacman.rect.left,
            self.Pacman.rect.top,
            self.Pacman.change_x,
            self.Pacman.change_y,
        ]
        min_distance = 100000
        for monster in self.Monsters:
            obs += [
                monster.rect.left,
                monster.rect.top,
                monster.change_x,
                monster.change_y,
            ]
            distance = np.sqrt(
                (self.Pacman.rect.left - monster.rect.left) ** 2
                + (self.Pacman.rect.top - monster.rect.top) ** 2
            )
            if distance < min_distance:
                min_distance = distance
        obs += [min_distance]
        if prev_obs is not None:
            n_blocks = prev_obs[-1]
            obs += [n_blocks - len(self.blocks_hit_list)]
            # blocks_obs = prev_obs[4 + 4 * len(self.monsters) :]
            # for block in self.blocks_hit_list:
            #     block_number = self.blocks_x_y_to_index[(block.rect.x, block.rect.y)]
            #     blocks_obs[block_number] = 0
            # obs = np.concatenate([np.array(obs), blocks_obs])
        else:
            # obs += len(self.block_list) * [1]
            obs += [len(self.block_list)]
        return tuple(np.array(obs).reshape(1, -1)[0])

    def get_action_space(self):
        return np.arange(len(self.ACTIONS_MAP))

    def get_state_size(self):
        return len(self.get_observation())

    def step(self, action):
        done = False
        info = {"status": "playing"}
        if not self.ai:
            events = pygame.event.get()
        else:
            events = [
                Event(self.ACTIONS_MAP[action]),
            ]
        for event in events:
            if not self.ai:
                if event.type == pygame.QUIT:
                    done = True
                    info = {"status": "quit"}

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.Pacman.changespeed(-30, 0)
                    if event.key == pygame.K_RIGHT:
                        self.Pacman.changespeed(30, 0)
                    if event.key == pygame.K_UP:
                        self.Pacman.changespeed(0, -30)
                    if event.key == pygame.K_DOWN:
                        self.Pacman.changespeed(0, 30)

                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT:
                        self.Pacman.changespeed(30, 0)
                    if event.key == pygame.K_RIGHT:
                        self.Pacman.changespeed(-30, 0)
                    if event.key == pygame.K_UP:
                        self.Pacman.changespeed(0, 30)
                    if event.key == pygame.K_DOWN:
                        self.Pacman.changespeed(0, -30)
            else:
                if event.key == pygame.K_LEFT:
                    self.Pacman.change_x = -30
                    self.Pacman.change_y = 0
                if event.key == pygame.K_RIGHT:
                    self.Pacman.change_x = 30
                    self.Pacman.change_y = 0
                if event.key == pygame.K_UP:
                    self.Pacman.change_x = 0
                    self.Pacman.change_y = -30
                if event.key == pygame.K_DOWN:
                    self.Pacman.change_x = 0
                    self.Pacman.change_y = 30

        self.Pacman.update(self.wall_list, self.gate)

        for monster, monster_details in zip(self.Monsters, self.monsters):
            returned = monster.changespeed(
                monster_details.directions,
                False,
                monster_details.turn,
                monster_details.steps,
                monster_details.l,
            )
            monster_details.turn = returned[0]
            monster_details.steps = returned[1]
            monster.changespeed(
                monster_details.directions,
                False,
                monster_details.turn,
                monster_details.steps,
                monster_details.l,
            )
            monster.update(self.wall_list, False)

        self.blocks_hit_list = pygame.sprite.spritecollide(
            self.Pacman, self.block_list, True
        )

        if len(self.blocks_hit_list) > 0:
            self.score += len(self.blocks_hit_list)

        if self.gui:
            self.screen.fill(black)

            self.wall_list.draw(self.screen)
            self.gate.draw(self.screen)
            self.all_sprites_list.draw(self.screen)
            self.monsta_list.draw(self.screen)

            text = self.font.render(
                "Score: " + str(self.score) + "/" + str(self.bll), True, red
            )
            self.screen.blit(text, [10, 10])

        if self.prev_score is None:
            self.prev_score = 0
        reward = self.score - self.prev_score
        self.prev_score = self.score

        if self.score == self.bll:
            done = True
            reward += 200
            info = {"status": "won"}

        monsta_hit_list = pygame.sprite.spritecollide(
            self.Pacman, self.monsta_list, False
        )

        if monsta_hit_list:
            done = True
            reward -= 100
            info = {"status": "lost"}

        if self.gui:
            pygame.display.flip()

        if not self.ai:
            self.clock.tick(10)
        self.obs = self.get_observation(self.obs)

        return self.obs, reward, done, info