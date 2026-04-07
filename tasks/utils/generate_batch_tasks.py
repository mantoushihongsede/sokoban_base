import json
from typing import List, Tuple

from core.state import SokobanMap, SokobanState
from core.env import SokobanEnv
from tasks.legality_tasks import ActionLegalityTask
from tasks.transition_tasks import StateTransitionTask
from tasks.utils.task_case_utils import classify_case

ACTIONS = ["up", "down", "left", "right"]

def load_envs_from_json(json_path: str) -> List[SokobanEnv]:
    """从本地 JSON 文件加载关卡并转换为 SokobanEnv 对象列表"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    envs = []
    for level_data in data:
        # JSON 中的坐标是 list，需要转换为 tuple，并用 frozenset 包装以符合底层类要求
        walls = frozenset(tuple(pos) for pos in level_data["walls"])
        targets = frozenset(tuple(pos) for pos in level_data["targets"])
        boxes = frozenset(tuple(pos) for pos in level_data["boxes"])
        agent_pos = tuple(level_data["agent_pos"])

        soko_map = SokobanMap(
            level_id=level_data["level_id"],
            height=level_data["height"],
            width=level_data["width"],
            walls=walls,
            targets=targets
        )

        soko_state = SokobanState(
            agent_pos=agent_pos,
            boxes=boxes
        )

        envs.append(SokobanEnv(soko_map, soko_state))

    return envs


def generate_legality_tasks(json_path: str, num_samples: int = 20):
    task_builder = ActionLegalityTask()
    # 替换原有的 build_demo_envs()
    envs = load_envs_from_json(json_path)

    tasks = []
    sample_id = 0

    for env in envs:
        for action in ACTIONS:
            if len(tasks) >= num_samples:
                return tasks

            case_type = classify_case(env, action)
            instance = task_builder.build(
                task_id=f"legality_{sample_id}",
                sokoban_map=env.map,
                state=env.state,
                candidate_action=action,
                case_type=case_type,
            )
            tasks.append((task_builder, instance))
            sample_id += 1


    return tasks


def generate_transition_tasks(json_path: str, num_samples: int = 20, legal_only: bool = True):
    task_builder = StateTransitionTask()
    # 替换原有的 build_demo_envs()
    envs = load_envs_from_json(json_path)

    tasks = []
    sample_id = 0

    for env in envs:
        for action in ACTIONS:
            if len(tasks) >= num_samples:
                return tasks

            if legal_only and (not env.is_legal_move(action)):
                continue

            case_type = classify_case(env, action)
            instance = task_builder.build(
                task_id=f"transition_{sample_id}",
                env=env,
                action=action,
                case_type=case_type,
            )
            tasks.append((task_builder, instance))
            sample_id += 1

    return tasks