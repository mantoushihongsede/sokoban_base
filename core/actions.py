from typing import Literal, Tuple

Action = Literal["up", "down", "left", "right"]

ACTION_MAP = {
"up": (-1, 0),
"down": (1, 0),
"left": (0, -1),
"right": (0, 1),
}

def move_pos(pos: Tuple[int, int], action: Action) -> Tuple[int, int]:
    dr, dc = ACTION_MAP[action]
    return (pos[0] + dr, pos[1] + dc)