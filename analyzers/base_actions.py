import collections
from typing import Any, Deque, Dict, List, Set, Tuple

from core.state import SokobanMap, SokobanState, Pos
from .json_utils import action_to_json, state_to_json


Direction = Tuple[int, int]


def directions() -> List[Direction]:
    return [(-1, 0), (1, 0), (0, -1), (0, 1)]


def in_bounds(sokoban_map: SokobanMap, pos: Pos) -> bool:
    return sokoban_map.in_bounds(pos)


def is_floor(sokoban_map: SokobanMap, pos: Pos) -> bool:
    return sokoban_map.in_bounds(pos) and pos not in sokoban_map.walls


def is_free_floor(sokoban_map: SokobanMap, pos: Pos, box_set: Set[Pos]) -> bool:
    return is_floor(sokoban_map, pos) and pos not in box_set


def player_reachable_tiles(sokoban_map: SokobanMap, state: SokobanState) -> Dict[str, Any]:
    box_set = set(state.boxes)
    start = state.agent_pos

    if not is_free_floor(sokoban_map, start, box_set):
        return {
            "player_pos": list(start),
            "reachable_tiles": [],
            "count": 0,
        }

    visited: Set[Pos] = {start}
    q: Deque[Pos] = collections.deque([start])

    while q:
        curr = q.popleft()
        cr, cc = curr
        for dr, dc in directions():
            nxt = (cr + dr, cc + dc)
            if nxt in visited:
                continue
            if not is_free_floor(sokoban_map, nxt, box_set):
                continue
            visited.add(nxt)
            q.append(nxt)

    reachable = sorted(visited)
    return {
        "player_pos": list(start),
        "reachable_tiles": [list(p) for p in reachable],
        "count": len(reachable),
    }


def list_legal_pushes(sokoban_map: SokobanMap, state: SokobanState) -> Dict[str, Any]:
    box_set = set(state.boxes)
    reachable_info = player_reachable_tiles(sokoban_map, state)
    reachable: Set[Pos] = {tuple(p) for p in reachable_info["reachable_tiles"]}

    actions: List[Dict[str, Any]] = []

    for box in sorted(box_set):
        br, bc = box
        for dr, dc in directions():
            player_from = (br - dr, bc - dc)
            box_to = (br + dr, bc + dc)

            if player_from not in reachable:
                continue
            if not is_free_floor(sokoban_map, box_to, box_set):
                continue

            actions.append(
                action_to_json(
                    box_from=box,
                    box_to=box_to,
                    player_from=player_from,
                    direction=(dr, dc),
                )
            )

    actions.sort(
        key=lambda a: (
            a["box_from"][0], a["box_from"][1],
            a["box_to"][0], a["box_to"][1],
            a["player_from"][0], a["player_from"][1],
            a["direction"][0], a["direction"][1],
        )
    )

    return {
        "legal_pushes": actions,
        "count": len(actions),
    }


def state_has_legal_push(sokoban_map: SokobanMap, state: SokobanState) -> Dict[str, Any]:
    pushes = list_legal_pushes(sokoban_map, state)
    return {
        "has_legal_push": pushes["count"] > 0,
        "legal_push_count": pushes["count"],
    }


def apply_push(
    sokoban_map: SokobanMap,
    state: SokobanState,
    action: Dict[str, Any],
) -> Dict[str, Any]:
    from .json_utils import action_json_to_tuple, normalize_action_json

    normalized_action = normalize_action_json(action)
    legal_actions = list_legal_pushes(sokoban_map, state)["legal_pushes"]

    if normalized_action not in legal_actions:
        return {
            "ok": False,
            "error": "Illegal push action.",
            "input_action": normalized_action,
        }

    box_from, box_to, _player_from, _direction = action_json_to_tuple(normalized_action)

    new_boxes = set(state.boxes)
    new_boxes.remove(box_from)
    new_boxes.add(box_to)

    new_state = SokobanState(
        agent_pos=box_from,
        boxes=frozenset(new_boxes),
    )

    return {
        "ok": True,
        "new_state": state_to_json(new_state),
    }