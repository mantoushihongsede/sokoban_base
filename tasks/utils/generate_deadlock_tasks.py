import json
from typing import Dict, List, Optional, Tuple

from core.state import SokobanMap, SokobanState
from core.env import SokobanEnv
from tasks.deadlock_tasks import (
    DefiniteDeadlockDetectionTask,
    StaticDeadSquaresTask,
    BoxStatusExplanationTask,
)

def load_records_from_json(json_path: str, gt_name: str) -> List[Dict]:
    """
    从本地 JSON 文件加载关卡并转换为 SokobanEnv 对象列表。

    兼容字段：
    - level_id / id
    - walls / wall_positions
    - targets / target_positions
    - boxes / box_positions
    - agent_pos / player_pos
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records: List[Dict] = []

    for level_data in data:
        level_id = level_data.get("level_id")

        walls_raw = level_data.get("walls")
        targets_raw = level_data.get("targets")
        boxes_raw = level_data.get("boxes")
        agent_pos_raw = level_data.get("agent_pos")

        if agent_pos_raw is None:
            raise ValueError(f"Level {level_id} missing agent_pos/player_pos field.")

        walls = frozenset(tuple(pos) for pos in walls_raw)
        targets = frozenset(tuple(pos) for pos in targets_raw)
        boxes = frozenset(tuple(pos) for pos in boxes_raw)
        agent_pos = tuple(agent_pos_raw)
        gt = level_data.get(gt_name, None)

        soko_map = SokobanMap(
            level_id=level_id,
            height=int(level_data["height"]),
            width=int(level_data["width"]),
            walls=walls,
            targets=targets,
        )

        soko_state = SokobanState(
            agent_pos=agent_pos,
            boxes=boxes,
        )

        records.append({
            "env": SokobanEnv(soko_map, soko_state),
            gt_name: gt,
        })

    return records


def generate_definite_deadlock_tasks(
    json_path: str,
    num_samples: int = 20,
) -> List[Tuple[DefiniteDeadlockDetectionTask, object]]:
    task_builder = DefiniteDeadlockDetectionTask()
    records = load_records_from_json(json_path, "ground_truth_deadlock") #1

    tasks = []
    sample_id = 0

    for record in records: #2
        if len(tasks) >= num_samples:
            return tasks

        gt = record.get("ground_truth_deadlock", None) #3
        env = record.get("env", None) #4

        instance = task_builder.build(
            task_id=f"deadlock_{sample_id}",
            sokoban_map=env.map,
            state=env.state,
            case_type="definite_deadlock",
            gt=gt, #5
        )
        tasks.append((task_builder, instance))
        sample_id += 1

    return tasks


def generate_static_dead_squares_tasks(
    json_path: str,
    num_samples: int = 20,
) -> List[Tuple[StaticDeadSquaresTask, object]]:
    task_builder = StaticDeadSquaresTask()
    records = load_records_from_json(json_path, "ground_truth_static_dead_squares") #1

    tasks = []
    sample_id = 0

    for record in records: #2
        if len(tasks) >= num_samples:
            return tasks

        gt = record.get("ground_truth_static_dead_squares", None) #3
        env = record.get("env", None) #4

        instance = task_builder.build(
            task_id=f"static_dead_{sample_id}",
            sokoban_map=env.map,
            state=env.state,
            case_type="static_dead_squares",
            gt=gt, #5
        )
        tasks.append((task_builder, instance))
        sample_id += 1

    return tasks


def generate_box_status_explanation_tasks(
    json_path: str,
    num_samples: int = 20,
) -> List[Tuple[BoxStatusExplanationTask, object]]:
    task_builder = BoxStatusExplanationTask()
    records = load_records_from_json(json_path, "ground_truth_box_status")

    tasks = []
    sample_id = 0

    for record in records:
        if len(tasks) >= num_samples:
            return tasks

        gt = record.get("ground_truth_box_status", None)
        env = record.get("env", None)

        if env is None or gt is None or not isinstance(gt, list) or len(gt) == 0:
            continue

        # 直接使用 ground_truth_box_status 中的所有箱子
        box_statuses = []
        for entry in gt:
            if not isinstance(entry, dict):
                continue
            if "box" not in entry:
                continue

            box_statuses.append({
                "box": tuple(entry["box"]),
                "on_target": bool(entry.get("on_target", False)),
                "on_static_dead_square": bool(entry.get("on_static_dead_square", False)),
                "in_blocked_2x2": bool(entry.get("in_blocked_2x2", False)),
                "currently_immovable": bool(entry.get("currently_immovable", False)),
                "legal_pushes": entry.get("legal_pushes", []),
            })

        if not box_statuses:
            continue

        instance = task_builder.build(
            task_id=f"box_status_{sample_id}",
            sokoban_map=env.map,
            state=env.state,
            case_type="box_status_explanation",
        )

        instance.metadata["ground_truth"] = box_statuses
        instance.metadata["num_boxes"] = len(box_statuses)

        tasks.append((task_builder, instance))
        sample_id += 1

    return tasks