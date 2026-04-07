from typing import Any, Dict, List

from core.state import SokobanMap, SokobanState, Pos
from .base_actions import directions, is_free_floor, list_legal_pushes
from .definite_deadlocks import DefiniteDeadlockAnalyzer


def explain_box_status(
    sokoban_map: SokobanMap,
    state: SokobanState,
    box: Pos,
    deadlock_analyzer: DefiniteDeadlockAnalyzer,
) -> Dict[str, Any]:
    if box not in state.boxes:
        return {
            "ok": False,
            "error": f"Position {list(box)} is not a box in the current state.",
        }

    legal_pushes = list_legal_pushes(sokoban_map, state)["legal_pushes"]
    box_pushes = [
        a for a in legal_pushes
        if tuple(a["box_from"]) == box
    ]

    dead2x2_boxes = {
        (p[0], p[1])
        for p in deadlock_analyzer.find_2x2_deadlocks(sokoban_map, state)["deadlocked_boxes"]
    }

    box_set = set(state.boxes)
    free_neighbors = 0
    br, bc = box
    for dr, dc in directions():
        nxt = (br + dr, bc + dc)
        if is_free_floor(sokoban_map, nxt, box_set):
            free_neighbors += 1

    on_static_dead_square = deadlock_analyzer.is_static_dead_square(sokoban_map, box)

    warnings: List[str] = []
    if on_static_dead_square:
        warnings.append("This box is on a static dead square.")
    if box in dead2x2_boxes:
        warnings.append("This box is part of a blocked 2x2 structure.")
    if len(box_pushes) == 0 and box not in sokoban_map.targets:
        warnings.append("This box is currently immovable.")
    if free_neighbors <= 1 and box not in sokoban_map.targets:
        warnings.append("This box has low local mobility.")

    return {
        "ok": True,
        "box": list(box),
        "on_target": box in sokoban_map.targets,
        "on_static_dead_square": on_static_dead_square,
        "in_blocked_2x2": box in dead2x2_boxes,
        "local_free_neighbor_count": free_neighbors,
        "legal_pushes": box_pushes,
        "warning_messages": sorted(set(warnings)),
    }