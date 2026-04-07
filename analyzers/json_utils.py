from typing import Any, Dict, List, Tuple

from core.state import SokobanMap, SokobanState, Pos
from core.serializer import serialize_state, serialize_full_state


Direction = Tuple[int, int]


def pos_to_json(pos: Pos) -> List[int]:
    return [pos[0], pos[1]]


def positions_to_json(positions: List[Pos]) -> List[List[int]]:
    return [pos_to_json(p) for p in positions]


def direction_to_json(direction: Direction) -> List[int]:
    return [direction[0], direction[1]]


def action_to_json(
    box_from: Pos,
    box_to: Pos,
    player_from: Pos,
    direction: Direction,
) -> Dict[str, Any]:
    return {
        "box_from": pos_to_json(box_from),
        "box_to": pos_to_json(box_to),
        "player_from": pos_to_json(player_from),
        "direction": direction_to_json(direction),
    }


def normalize_action_json(action: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "box_from": [int(action["box_from"][0]), int(action["box_from"][1])],
        "box_to": [int(action["box_to"][0]), int(action["box_to"][1])],
        "player_from": [int(action["player_from"][0]), int(action["player_from"][1])],
        "direction": [int(action["direction"][0]), int(action["direction"][1])],
    }


def action_json_to_tuple(action: Dict[str, Any]) -> Tuple[Pos, Pos, Pos, Direction]:
    normalized = normalize_action_json(action)
    box_from = (normalized["box_from"][0], normalized["box_from"][1])
    box_to = (normalized["box_to"][0], normalized["box_to"][1])
    player_from = (normalized["player_from"][0], normalized["player_from"][1])
    direction = (normalized["direction"][0], normalized["direction"][1])
    return box_from, box_to, player_from, direction


def state_to_json(state: SokobanState) -> Dict[str, Any]:
    return serialize_state(state)


def full_state_to_json(sokoban_map: SokobanMap, state: SokobanState) -> Dict[str, Any]:
    return serialize_full_state(sokoban_map, state)


def json_to_pos(pos_json: List[int]) -> Pos:
    return (int(pos_json[0]), int(pos_json[1]))


def json_to_positions(items: List[List[int]]) -> List[Pos]:
    return [json_to_pos(x) for x in items]


def json_to_state(state_json: Dict[str, Any]) -> SokobanState:
    return SokobanState(
        agent_pos=json_to_pos(state_json["player_pos"]),
        boxes=frozenset(json_to_positions(state_json["box_positions"])),
    )


def json_to_map(map_json: Dict[str, Any]) -> SokobanMap:
    return SokobanMap(
        level_id=map_json["id"],
        height=int(map_json["height"]),
        width=int(map_json["width"]),
        walls=frozenset(json_to_positions(map_json["wall_positions"])),
        targets=frozenset(json_to_positions(map_json["target_positions"])),
    )