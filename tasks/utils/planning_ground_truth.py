from typing import Any, Dict, List, Tuple

from core.state import SokobanMap, SokobanState, Pos
from .planning_case_utils import (
    manhattan,
    sort_boxes,
    count_box_push_options,
    is_corner_dead_square,
    ALLOWED_PHASES,
)


def heuristic_box_scores(sokoban_map: SokobanMap, state: SokobanState) -> List[Dict[str, Any]]:
    """
    分数越低越优先。
    """
    rows = []
    for box in sort_boxes(state.boxes):
        min_target_dist = min(manhattan(box, t) for t in sokoban_map.targets) if sokoban_map.targets else 999
        on_target = box in sokoban_map.targets
        push_options = count_box_push_options(sokoban_map, state, box)
        corner_risk = 1 if is_corner_dead_square(sokoban_map, box) else 0

        score = 0
        score += min_target_dist
        score += 4 * corner_risk
        score += 3 if on_target else 0
        score += 2 * max(0, 2 - push_options)

        rows.append({
            "box": box,
            "score": score,
            "min_target_dist": min_target_dist,
            "push_options": push_options,
            "corner_risk": corner_risk,
            "on_target": on_target,
        })
    rows.sort(key=lambda x: (x["score"], x["min_target_dist"], x["box"]))
    return rows


def gold_box_priority_order(sokoban_map: SokobanMap, state: SokobanState) -> List[Pos]:
    return [row["box"] for row in heuristic_box_scores(sokoban_map, state)]


def greedy_box_target_assignment(sokoban_map: SokobanMap, state: SokobanState) -> List[Dict[str, Any]]:
    """
    简单一对一 greedy matching，足够作为 benchmark heuristic gold。
    """
    remaining_targets = set(sokoban_map.targets)
    assignments = []

    for box in gold_box_priority_order(sokoban_map, state):
        if not remaining_targets:
            break
        best_target = min(
            remaining_targets,
            key=lambda t: (manhattan(box, t), t)
        )
        assignments.append({
            "box": list(box),
            "target": list(best_target),
            "reason": f"heuristic nearest-target assignment, distance={manhattan(box, best_target)}"
        })
        remaining_targets.remove(best_target)

    return assignments


def infer_phase(sokoban_map: SokobanMap, state: SokobanState) -> str:
    boxes_not_on_target = [b for b in state.boxes if b not in sokoban_map.targets]
    if not boxes_not_on_target:
        return "deliver_box"

    top_rows = heuristic_box_scores(sokoban_map, state)
    if not top_rows:
        return "setup"

    best = top_rows[0]
    if best["corner_risk"] == 1 or best["push_options"] == 0:
        return "clear_path"
    if best["min_target_dist"] <= 2 and best["push_options"] >= 1:
        return "deliver_box"
    if best["push_options"] >= 2:
        return "setup"
    return "reposition"


def discover_candidate_subgoals(sokoban_map: SokobanMap, state: SokobanState) -> List[Dict[str, Any]]:
    rows = heuristic_box_scores(sokoban_map, state)
    subgoals = []

    priority = 1
    for row in rows[:3]:
        box = row["box"]
        nearest_target = min(sokoban_map.targets, key=lambda t: (manhattan(box, t), t)) if sokoban_map.targets else None

        if row["push_options"] == 0:
            subgoals.append({
                "type": "clear_path",
                "object": list(box),
                "priority": priority,
            })
            priority += 1
            continue

        if row["corner_risk"] == 1 and box not in sokoban_map.targets:
            subgoals.append({
                "type": "avoid_deadlock",
                "object": list(box),
                "priority": priority,
            })
            priority += 1

        if nearest_target is not None and box not in sokoban_map.targets:
            subgoals.append({
                "type": "deliver_box",
                "object": list(box),
                "target": list(nearest_target),
                "priority": priority,
            })
            priority += 1
        else:
            subgoals.append({
                "type": "reposition_box",
                "object": list(box),
                "priority": priority,
            })
            priority += 1

    return subgoals[:5]


def order_subproblems(candidate_subgoals: List[Dict[str, Any]]) -> List[str]:
    """
    简单规则：
    clear_path / avoid_deadlock 优先于 deliver_box / reposition_box / approach_box
    """
    priority_map = {
        "avoid_deadlock": 0,
        "clear_path": 1,
        "free_target": 2,
        "approach_box": 3,
        "reposition_box": 4,
        "deliver_box": 5,
    }

    decorated = []
    for i, sg in enumerate(candidate_subgoals):
        sg_id = sg["id"]
        sg_type = sg.get("type", "")
        decorated.append((priority_map.get(sg_type, 999), i, sg_id))

    decorated.sort()
    return [x[2] for x in decorated]


def choose_better_horizon_option(option_a: Dict[str, Any], option_b: Dict[str, Any]) -> str:
    """
    输入假设中带有 heuristic_risk / heuristic_progress 字段。
    风险优先级高于短期收益。
    """
    a_risk = option_a.get("heuristic_risk", 0)
    b_risk = option_b.get("heuristic_risk", 0)
    a_prog = option_a.get("heuristic_progress", 0)
    b_prog = option_b.get("heuristic_progress", 0)

    if a_risk != b_risk:
        return "A" if a_risk < b_risk else "B"
    return "A" if a_prog >= b_prog else "B"