from .game import PacManEnv, PlayerPos, Monster, Gate

walls = [
    [0, 0, 6, 600],
    [0, 0, 600, 6],
    [0, 600, 606, 6],
    [600, 0, 6, 606],
    [300, 0, 6, 66],
    [60, 60, 186, 6],
    [360, 60, 186, 6],
    [60, 120, 66, 6],
    [60, 120, 6, 126],
    [180, 120, 246, 6],
    [300, 120, 6, 66],
    [480, 120, 66, 6],
    [540, 120, 6, 126],
    [120, 180, 126, 6],
    [120, 180, 6, 126],
    [360, 180, 126, 6],
    [480, 180, 6, 126],
    [180, 240, 6, 126],
    [180, 360, 246, 6],
    [420, 240, 6, 126],
    [240, 240, 42, 6],
    [324, 240, 42, 6],
    [240, 240, 6, 66],
    [240, 300, 126, 6],
    [360, 240, 6, 66],
    [0, 300, 66, 6],
    [540, 300, 66, 6],
    [60, 360, 66, 6],
    [60, 360, 6, 186],
    [480, 360, 66, 6],
    [540, 360, 6, 186],
    [120, 420, 366, 6],
    [120, 420, 6, 66],
    [480, 420, 6, 66],
    [180, 480, 246, 6],
    [300, 480, 6, 66],
    [120, 540, 126, 6],
    [360, 540, 126, 6],
]

Pinky_directions = [
    [0, -30, 4],
    [15, 0, 9],
    [0, 15, 11],
    [-15, 0, 23],
    [0, 15, 7],
    [15, 0, 3],
    [0, -15, 3],
    [15, 0, 19],
    [0, 15, 3],
    [15, 0, 3],
    [0, 15, 3],
    [15, 0, 3],
    [0, -15, 15],
    [-15, 0, 7],
    [0, 15, 3],
    [-15, 0, 19],
    [0, -15, 11],
    [15, 0, 9],
]

Blinky_directions = [
    [0, -15, 4],
    [15, 0, 9],
    [0, 15, 11],
    [15, 0, 3],
    [0, 15, 7],
    [-15, 0, 11],
    [0, 15, 3],
    [15, 0, 15],
    [0, -15, 15],
    [15, 0, 3],
    [0, -15, 11],
    [-15, 0, 3],
    [0, -15, 11],
    [-15, 0, 3],
    [0, -15, 3],
    [-15, 0, 7],
    [0, -15, 3],
    [15, 0, 15],
    [0, 15, 15],
    [-15, 0, 3],
    [0, 15, 3],
    [-15, 0, 3],
    [0, -15, 7],
    [-15, 0, 3],
    [0, 15, 7],
    [-15, 0, 11],
    [0, -15, 7],
    [15, 0, 5],
]

Inky_directions = [
    [30, 0, 2],
    [0, -15, 4],
    [15, 0, 10],
    [0, 15, 7],
    [15, 0, 3],
    [0, -15, 3],
    [15, 0, 3],
    [0, -15, 15],
    [-15, 0, 15],
    [0, 15, 3],
    [15, 0, 15],
    [0, 15, 11],
    [-15, 0, 3],
    [0, -15, 7],
    [-15, 0, 11],
    [0, 15, 3],
    [-15, 0, 11],
    [0, 15, 7],
    [-15, 0, 3],
    [0, -15, 3],
    [-15, 0, 3],
    [0, -15, 15],
    [15, 0, 15],
    [0, 15, 3],
    [-15, 0, 15],
    [0, 15, 11],
    [15, 0, 3],
    [0, -15, 11],
    [15, 0, 11],
    [0, 15, 3],
    [15, 0, 1],
]

Clyde_directions = [
    [-30, 0, 2],
    [0, -15, 4],
    [15, 0, 5],
    [0, 15, 7],
    [-15, 0, 11],
    [0, -15, 7],
    [-15, 0, 3],
    [0, 15, 7],
    [-15, 0, 7],
    [0, 15, 15],
    [15, 0, 15],
    [0, -15, 3],
    [-15, 0, 11],
    [0, -15, 7],
    [15, 0, 3],
    [0, -15, 11],
    [15, 0, 9],
]

w = 303 - 16  # Width
p_h = (7 * 60) + 19  # Pacman height
m_h = (4 * 60) + 19  # Monster height
b_h = (3 * 60) + 19  # Binky height
i_w = 303 - 16 - 32  # Inky width
c_w = 303 + (32 - 16)  # Clyde width

from copy import deepcopy

gate = Gate(282, 242, 42, 2)
pacman_player = PlayerPos(w, p_h)
blinky = Monster(w, m_h, name="Blinky", directions=deepcopy(Blinky_directions))
pinky = Monster(w, b_h, name="Pinky", directions=deepcopy(Pinky_directions))
inky = Monster(i_w, m_h, name="Inky", directions=deepcopy(Inky_directions))
clyde = Monster(c_w, m_h, name="Clyde", directions=deepcopy(Clyde_directions))
monsters = [blinky, pinky, inky, clyde]

base_level = {
    "walls": walls,
    "gate": gate,
    "player": pacman_player,
    "monsters": monsters,
    "height": 606,
    "width": 606,
}

def get_level_1():
    walls_2 = [
        [0, 0, 6, 600],
        [0, 0, 600, 6],
        [0, 600, 606, 6],
        [600, 0, 6, 606],
        [240, 240, 42, 6],
        [324, 240, 42, 6],
        [240, 240, 6, 66],
        [240, 300, 126, 6],
        [360, 240, 6, 66],
    ]

    Pinky_directions = [
        [0, -30, 4],
        [15, 0, 9],
        [0, 15, 11],
        [-15, 0, 23],
        [0, 15, 7],
        [15, 0, 3],
        [0, -15, 3],
        [15, 0, 19],
        [0, 15, 3],
        [15, 0, 3],
        [0, 15, 3],
        [15, 0, 3],
        [0, -15, 15],
        [-15, 0, 7],
        [0, 15, 3],
        [-15, 0, 19],
        [0, -15, 11],
        [15, 0, 9],
    ]

    Blinky_directions = [
        [0, -15, 4],
        [15, 0, 9],
        [0, 15, 11],
        [15, 0, 3],
        [0, 15, 7],
        [-15, 0, 11],
        [0, 15, 3],
        [15, 0, 15],
        [0, -15, 15],
        [15, 0, 3],
        [0, -15, 11],
        [-15, 0, 3],
        [0, -15, 11],
        [-15, 0, 3],
        [0, -15, 3],
        [-15, 0, 7],
        [0, -15, 3],
        [15, 0, 15],
        [0, 15, 15],
        [-15, 0, 3],
        [0, 15, 3],
        [-15, 0, 3],
        [0, -15, 7],
        [-15, 0, 3],
        [0, 15, 7],
        [-15, 0, 11],
        [0, -15, 7],
        [15, 0, 5],
    ]

    Inky_directions = [
        [30, 0, 2],
        [0, -15, 4],
        [15, 0, 10],
        [0, 15, 7],
        [15, 0, 3],
        [0, -15, 3],
        [15, 0, 3],
        [0, -15, 15],
        [-15, 0, 15],
        [0, 15, 3],
        [15, 0, 15],
        [0, 15, 11],
        [-15, 0, 3],
        [0, -15, 7],
        [-15, 0, 11],
        [0, 15, 3],
        [-15, 0, 11],
        [0, 15, 7],
        [-15, 0, 3],
        [0, -15, 3],
        [-15, 0, 3],
        [0, -15, 15],
        [15, 0, 15],
        [0, 15, 3],
        [-15, 0, 15],
        [0, 15, 11],
        [15, 0, 3],
        [0, -15, 11],
        [15, 0, 11],
        [0, 15, 3],
        [15, 0, 1],
    ]

    Clyde_directions = [
        [-30, 0, 2],
        [0, -15, 4],
        [15, 0, 5],
        [0, 15, 7],
        [-15, 0, 11],
        [0, -15, 7],
        [-15, 0, 3],
        [0, 15, 7],
        [-15, 0, 7],
        [0, 15, 15],
        [15, 0, 15],
        [0, -15, 3],
        [-15, 0, 11],
        [0, -15, 7],
        [15, 0, 3],
        [0, -15, 11],
        [15, 0, 9],
    ]

    w = 303 - 16  # Width
    p_h = (7 * 60) + 19  # Pacman height
    m_h = (4 * 60) + 19  # Monster height
    b_h = (3 * 60) + 19  # Binky height
    i_w = 303 - 16 - 32  # Inky width
    c_w = 303 + (32 - 16)  # Clyde width

    from copy import deepcopy

    gate = Gate(282, 242, 42, 2)
    pacman_player = PlayerPos(w, p_h)
    blinky = Monster(w, m_h, name="Blinky", directions=deepcopy(Blinky_directions))
    pinky = Monster(w, b_h, name="Pinky", directions=deepcopy(Pinky_directions))
    inky = Monster(i_w, m_h, name="Inky", directions=deepcopy(Inky_directions))
    clyde = Monster(c_w, m_h, name="Clyde", directions=deepcopy(Clyde_directions))
    monsters = [pinky, inky]

    level_1 = {
        "walls": walls_2,
        "gate": gate,
        "player": pacman_player,
        "monsters": monsters,
        "height": 606,
        "width": 606,
    }

    return level_1
