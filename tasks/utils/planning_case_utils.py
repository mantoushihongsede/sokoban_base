from typing import Any, Dict, List, Optional, Tuple
from collections import deque

from core.state import SokobanMap, SokobanState, Pos


ALLOWED_SUBGOAL_TYPES = {
    "clear_path",
    "deliver_box",
    "reposition_box",
    "approach_box",
    "free_target",
    "avoid_deadlock",
}

ALLOWED_PHASES = {
    "clear_path",
    "deliver_box",
    "reposition",
    "setup",
    "recover",
}


def normalize_pos(pos: Any) -> Optional[Pos]:
    if not isinstance(pos, (list, tuple)) or len(pos) != 2:
        return None
    if not all(isinstance(x, int) for x in pos):
        return None
    return (pos[0], pos[1])


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def is_free_floor(sokoban_map: SokobanMap, state: SokobanState, pos: Pos) -> bool:
    return (
        sokoban_map.in_bounds(pos)
        and pos not in sokoban_map.walls
        and pos not in state.boxes
    )


def neighbors(pos: Pos) -> List[Pos]:
    r, c = pos
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def reachable_tiles_without_pushing(sokoban_map: SokobanMap, state: SokobanState) -> List[Pos]:
    start = state.agent_pos
    q = deque([start])
    visited = {start}

    while q:
        cur = q.popleft()
        for nxt in neighbors(cur):
            if nxt in visited:
                continue
            if not sokoban_map.in_bounds(nxt):
                continue
            if nxt in sokoban_map.walls:
                continue
            if nxt in state.boxes:
                continue
            visited.add(nxt)
            q.append(nxt)

    return sorted(visited)


def box_push_options(sokoban_map: SokobanMap, state: SokobanState, box: Pos) -> List[Dict[str, Any]]:
    """
    返回当前状态下，该箱子可能被推的方向（仅几何+可达性判定，不做深度规划）
    """
    reachable = set(reachable_tiles_without_pushing(sokoban_map, state))
    results = []
    r, c = box

    dirs = [
        ((-1, 0), (r + 1, c), (r - 1, c), "up"),
        ((1, 0), (r - 1, c), (r + 1, c), "down"),
        ((0, -1), (r, c + 1), (r, c - 1), "left"),
        ((0, 1), (r, c - 1), (r, c + 1), "right"),
    ]

    for (drdc, player_need, box_to, name) in dirs:
        if player_need not in reachable:
            continue
        if not sokoban_map.in_bounds(box_to):
            continue
        if box_to in sokoban_map.walls:
            continue
        if box_to in state.boxes:
            continue
        results.append({
            "direction": name,
            "player_need": player_need,
            "box_to": box_to,
        })

    return results


def count_box_push_options(sokoban_map: SokobanMap, state: SokobanState, box: Pos) -> int:
    return len(box_push_options(sokoban_map, state, box))


def is_corner_dead_square(sokoban_map: SokobanMap, pos: Pos) -> bool:
    """
    简单静态角落启发式：非目标位且相邻两侧是墙。
    注意：这是 heuristic，不是 sound full dead-square analysis。
    """
    if pos in sokoban_map.targets:
        return False
    r, c = pos
    up = (r - 1, c) in sokoban_map.walls
    down = (r + 1, c) in sokoban_map.walls
    left = (r, c - 1) in sokoban_map.walls
    right = (r, c + 1) in sokoban_map.walls
    return (up and left) or (up and right) or (down and left) or (down and right)


def sort_boxes(boxes) -> List[Pos]:
    return sorted(list(boxes))


def normalize_subgoal_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    sg_type = item.get("type")
    if sg_type not in ALLOWED_SUBGOAL_TYPES:
        return None

    out = {"type": sg_type}

    obj = item.get("object")
    if obj is not None:
        norm_obj = normalize_pos(obj)
        if norm_obj is None:
            return None
        out["object"] = list(norm_obj)

    target = item.get("target")
    if target is not None:
        norm_target = normalize_pos(target)
        if norm_target is None:
            return None
        out["target"] = list(norm_target)

    priority = item.get("priority")
    if not isinstance(priority, int):
        return None
    out["priority"] = priority
    return out


def pairwise_order_accuracy(pred_order: List[str], gold_order: List[str]) -> float:
    pred_index = {x: i for i, x in enumerate(pred_order)}
    gold = [x for x in gold_order if x in pred_index]
    if len(gold) < 2:
        return 1.0

    total = 0
    correct = 0
    for i in range(len(gold)):
        for j in range(i + 1, len(gold)):
            total += 1
            if pred_index[gold[i]] < pred_index[gold[j]]:
                correct += 1
    return correct / total if total > 0 else 1.0