from typing import Dict, Any, List
from .state import SokobanMap, SokobanState


def serialize_state(state: SokobanState) -> Dict[str, Any]:
    return {
        "player_pos": list(state.agent_pos),
        "box_positions": sorted([list(b) for b in state.boxes]),
    }


def serialize_full_state(sokoban_map: SokobanMap, state: SokobanState) -> Dict[str, Any]:
    return {
        "id": sokoban_map.level_id,
        "player_pos": list(state.agent_pos),
        "box_positions": sorted([list(b) for b in state.boxes]),
        "target_positions": sorted([list(t) for t in sokoban_map.targets]),
        "wall_positions": sorted([list(w) for w in sokoban_map.walls]),
        "is_solved": state.is_solved(sokoban_map),
    }


def render_text_map(sokoban_map: SokobanMap, state: SokobanState) -> str:
    grid = [["." for _ in range(sokoban_map.width)] for _ in range(sokoban_map.height)]

    for r, c in sokoban_map.walls:
        grid[r][c] = "#"

    for r, c in sokoban_map.targets:
        if grid[r][c] == ".":
            grid[r][c] = "T"

    for r, c in state.boxes:
        if grid[r][c] == "T":
            grid[r][c] = "*"
        else:
            grid[r][c] = "B"

    ar, ac = state.agent_pos
    if grid[ar][ac] == "T":
        grid[ar][ac] = "+"
    else:
        grid[ar][ac] = "A"

    return "\n".join(" ".join(row) for row in grid)