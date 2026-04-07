import json
from typing import Dict, List, Optional, Tuple
from core.state import SokobanMap, SokobanState
from core.env import SokobanEnv


def load_envs_from_json(json_path: str) -> List[SokobanEnv]:
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

    envs: List[SokobanEnv] = []

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

        envs.append(SokobanEnv(soko_map, soko_state))

    return envs


# 没啥用，我当前的数据集不包含元信息，用不着case结构
# def load_deadlock_cases(json_path: str) -> List[Dict]:
#     """
#     读取 deadlock_cases.json。

#     支持两种格式：
#     1. 顶层直接是 list
#     2. 顶层是 dict，包含 "cases" 字段
#     """
#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     if isinstance(data, list):
#         return data

#     if isinstance(data, dict) and "cases" in data and isinstance(data["cases"], list):
#         return data["cases"]

#     raise ValueError("Invalid deadlock cases JSON format. Expected list or {'cases': [...]}.")


# def _build_env_from_case(case: Dict) -> SokobanEnv:
#     """
#     从单个 case 构建 SokobanEnv。
#     case 必须至少包含地图和状态信息。
#     """
#     level_id = case.get("level_id", case.get("id", case.get("case_id", "unknown_case")))

#     walls_raw = case.get("walls", case.get("wall_positions", []))
#     targets_raw = case.get("targets", case.get("target_positions", []))
#     boxes_raw = case.get("boxes", case.get("box_positions", []))
#     agent_pos_raw = case.get("agent_pos", case.get("player_pos"))

#     if agent_pos_raw is None:
#         raise ValueError(f"Case {level_id} missing agent_pos/player_pos field.")

#     soko_map = SokobanMap(
#         level_id=level_id,
#         height=int(case["height"]),
#         width=int(case["width"]),
#         walls=frozenset(tuple(pos) for pos in walls_raw),
#         targets=frozenset(tuple(pos) for pos in targets_raw),
#     )

#     soko_state = SokobanState(
#         agent_pos=tuple(agent_pos_raw),
#         boxes=frozenset(tuple(pos) for pos in boxes_raw),
#     )

#     return SokobanEnv(soko_map, soko_state)