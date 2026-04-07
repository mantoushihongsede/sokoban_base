from typing import Dict, List, Optional, Tuple

from core.env import SokobanEnv
from core.state import Pos
from analyzers.tools import SokobanAnalysisTools


def has_static_dead_square(env: SokobanEnv) -> bool:
    tools = SokobanAnalysisTools()
    result = tools.get_static_dead_squares(env.map)
    return result.get("count", 0) > 0


def count_static_dead_squares(env: SokobanEnv) -> int:
    tools = SokobanAnalysisTools()
    result = tools.get_static_dead_squares(env.map)
    return int(result.get("count", 0))


def count_deadlocked_boxes(env: SokobanEnv) -> int:
    tools = SokobanAnalysisTools()
    result = tools.find_definite_deadlocks(env.map, env.state)
    return len(result.get("dead_positions", []))


def count_immovable_boxes(env: SokobanEnv) -> int:
    tools = SokobanAnalysisTools()
    result = tools.currently_immovable_boxes(env.map, env.state)
    return int(result.get("count", 0))


def list_box_case_labels(env: SokobanEnv) -> List[Dict]:
    """
    为当前状态中的每个箱子生成一个轻量标签，便于后续挑选 explanation case。
    """
    tools = SokobanAnalysisTools()

    deadlock_result = tools.find_definite_deadlocks(env.map, env.state)
    static_result = tools.boxes_on_static_dead_squares(env.map, env.state)
    dead2x2_result = tools.find_2x2_deadlocks(env.map, env.state)
    immovable_result = tools.currently_immovable_boxes(env.map, env.state)

    dead_positions = {tuple(p) for p in deadlock_result.get("dead_positions", [])}
    static_positions = {tuple(p) for p in static_result.get("boxes_on_static_dead_squares", [])}
    dead2x2_positions = {tuple(p) for p in dead2x2_result.get("deadlocked_boxes", [])}
    immovable_positions = {tuple(p) for p in immovable_result.get("immovable_boxes", [])}

    results = []
    for box in sorted(env.state.boxes):
        labels = []

        if box in env.map.targets:
            labels.append("on_target")
        else:
            labels.append("off_target")

        if box in static_positions:
            labels.append("static_dead_square")

        if box in dead2x2_positions:
            labels.append("blocked_2x2")

        if box in dead_positions:
            labels.append("definite_deadlock_box")

        if box in immovable_positions:
            labels.append("currently_immovable")

        explain = tools.explain_box_status(env.map, env.state, box)
        legal_push_count = len(explain.get("legal_pushes", []))

        if legal_push_count > 0:
            labels.append("pushable")
        else:
            labels.append("not_pushable")

        results.append({
            "box": list(box),
            "labels": labels,
            "legal_push_count": legal_push_count,
            "warning_messages": explain.get("warning_messages", []),
        })

    return results


def pick_representative_box(
    env: SokobanEnv,
    preferred_labels: Optional[List[str]] = None,
) -> Optional[Pos]:
    """
    从当前状态里挑一个“代表箱子”，优先顺序：
    1. definite deadlock box
    2. static dead square
    3. blocked 2x2
    4. currently immovable
    5. off_target pushable
    6. 任意箱子
    """
    box_infos = list_box_case_labels(env)

    if not box_infos:
        return None

    if preferred_labels:
        for label in preferred_labels:
            for info in box_infos:
                if label in info["labels"]:
                    box = info["box"]
                    return (box[0], box[1])

    priority_order = [
        "definite_deadlock_box",
        "static_dead_square",
        "blocked_2x2",
        "currently_immovable",
        "off_target",
        "pushable",
    ]

    for label in priority_order:
        for info in box_infos:
            if label in info["labels"]:
                box = info["box"]
                return (box[0], box[1])

    first_box = box_infos[0]["box"]
    return (first_box[0], first_box[1])


def classify_deadlock_case(env: SokobanEnv) -> str:
    """
    对整个状态进行粗分类，用于 batch 采样或数据集组织。
    """
    tools = SokobanAnalysisTools()

    if env.state.is_solved(env.map):
        return "solved"

    deadlock_result = tools.find_definite_deadlocks(env.map, env.state)
    if deadlock_result.get("is_deadlock", False):
        static_positions = {
            tuple(p) for p in tools.boxes_on_static_dead_squares(env.map, env.state).get(
                "boxes_on_static_dead_squares", []
            )
        }
        dead2x2_positions = {
            tuple(p) for p in tools.find_2x2_deadlocks(env.map, env.state).get(
                "deadlocked_boxes", []
            )
        }

        has_static = len(static_positions) > 0
        has_2x2 = len(dead2x2_positions) > 0

        if has_static and has_2x2:
            return "definite_deadlock_mixed"
        if has_static:
            return "definite_deadlock_static_dead_square"
        if has_2x2:
            return "definite_deadlock_blocked_2x2"
        return "definite_deadlock_other"

    warning_result = tools.get_heuristic_warnings(env.map, env.state)
    if warning_result.get("has_warning", False):
        reasons = set(warning_result.get("warning_reasons", []))

        if "The current state has no legal push actions." in reasons:
            return "warning_no_legal_push"

        return "warning_other"

    return "non_deadlock_normal"