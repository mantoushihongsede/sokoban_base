import json
from typing import List, Dict, Any

from core.state import SokobanMap, SokobanState
from core.env import SokobanEnv

from tasks.subgoal_discovery_tasks import CandidateSubgoalDiscoveryTask
from tasks.box_priority_tasks import BoxPriorityRankingTask
from tasks.assignment_tasks import BoxTargetAssignmentTask
from tasks.phase_tasks import PhaseRecognitionTask
from tasks.subproblem_ordering_tasks import SubproblemOrderingTask
from tasks.horizon_choice_tasks import LongShortHorizonChoiceTask

from tasks.utils.planning_ground_truth import (
    discover_candidate_subgoals,
    order_subproblems,
)
from tasks.utils.generate_batch_tasks import load_envs_from_json


def generate_subgoal_discovery_tasks(json_path: str, num_samples: int = 20):
    task_builder = CandidateSubgoalDiscoveryTask()
    envs = load_envs_from_json(json_path)

    tasks = []
    sample_id = 0
    idx = 0

    while len(tasks) < num_samples and envs:
        env = envs[idx % len(envs)]
        instance = task_builder.build(
            task_id=f"subgoal_discovery_{sample_id}",
            sokoban_map=env.map,
            state=env.state,
            source="generated_planning_batch_v1",
        )
        tasks.append((task_builder, instance))
        sample_id += 1
        idx += 1

    return tasks


def generate_box_priority_tasks(json_path: str, num_samples: int = 20):
    task_builder = BoxPriorityRankingTask()
    envs = load_envs_from_json(json_path)

    tasks = []
    sample_id = 0
    idx = 0

    while len(tasks) < num_samples and envs:
        env = envs[idx % len(envs)]
        instance = task_builder.build(
            task_id=f"box_priority_{sample_id}",
            sokoban_map=env.map,
            state=env.state,
            source="generated_planning_batch_v1",
        )
        tasks.append((task_builder, instance))
        sample_id += 1
        idx += 1

    return tasks


def generate_box_target_assignment_tasks(json_path: str, num_samples: int = 20):
    task_builder = BoxTargetAssignmentTask()
    envs = load_envs_from_json(json_path)

    tasks = []
    sample_id = 0
    idx = 0

    while len(tasks) < num_samples and envs:
        env = envs[idx % len(envs)]
        instance = task_builder.build(
            task_id=f"box_target_assignment_{sample_id}",
            sokoban_map=env.map,
            state=env.state,
            source="generated_planning_batch_v1",
        )
        tasks.append((task_builder, instance))
        sample_id += 1
        idx += 1

    return tasks


def generate_phase_recognition_tasks(json_path: str, num_samples: int = 20):
    task_builder = PhaseRecognitionTask()
    envs = load_envs_from_json(json_path)

    tasks = []
    sample_id = 0
    idx = 0

    while len(tasks) < num_samples and envs:
        env = envs[idx % len(envs)]
        instance = task_builder.build(
            task_id=f"phase_recognition_{sample_id}",
            sokoban_map=env.map,
            state=env.state,
            source="generated_planning_batch_v1",
        )
        tasks.append((task_builder, instance))
        sample_id += 1
        idx += 1

    return tasks


def _build_subproblem_candidates_from_env(env: SokobanEnv) -> List[Dict[str, Any]]:
    """
    用 heuristic subgoal 结果包装成带 id 的 candidate_subgoals，供 SubproblemOrderingTask 使用。
    """
    raw_subgoals = discover_candidate_subgoals(env.map, env.state)
    candidates = []

    for i, sg in enumerate(raw_subgoals):
        item = {
            "id": f"sg{i+1}",
            "type": sg["type"],
        }
        if "object" in sg:
            item["object"] = sg["object"]
        if "target" in sg:
            item["target"] = sg["target"]
        if "priority" in sg:
            item["priority"] = sg["priority"]
        candidates.append(item)

    return candidates


def generate_subproblem_ordering_tasks(json_path: str, num_samples: int = 20):
    task_builder = SubproblemOrderingTask()
    envs = load_envs_from_json(json_path)

    tasks = []
    sample_id = 0
    idx = 0

    while len(tasks) < num_samples and envs:
        env = envs[idx % len(envs)]
        candidates = _build_subproblem_candidates_from_env(env)

        # 至少两个子任务才有排序意义
        if len(candidates) >= 2:
            instance = task_builder.build(
                task_id=f"subproblem_ordering_{sample_id}",
                candidate_subgoals=candidates,
                level_id=env.map.level_id,
                source="generated_planning_batch_v1",
            )
            tasks.append((task_builder, instance))
            sample_id += 1

        idx += 1

        # 避免死循环：如果所有 env 都构不出足够候选任务
        if idx > len(envs) * 3 and not tasks:
            break

    return tasks


def _make_horizon_options_from_env(env: SokobanEnv) -> Dict[str, Any]:
    """
    生成 A/B 选项。
    这里先用启发式字段构造，不依赖更复杂求解器。
    """
    subgoals = discover_candidate_subgoals(env.map, env.state)

    if len(subgoals) >= 2:
        sg1 = subgoals[0]
        sg2 = subgoals[1]
    elif len(subgoals) == 1:
        sg1 = subgoals[0]
        sg2 = {
            "type": "reposition_box",
            "object": sg1.get("object"),
            "priority": sg1.get("priority", 1) + 1,
        }
    else:
        boxes = sorted(env.state.boxes)
        box = list(boxes[0]) if boxes else [0, 0]
        sg1 = {"type": "deliver_box", "object": box, "priority": 1}
        sg2 = {"type": "clear_path", "object": box, "priority": 2}

    # 简单风险/收益映射
    def sg_to_option(label: str, sg: Dict[str, Any]) -> Dict[str, Any]:
        sg_type = sg.get("type", "reposition_box")
        obj = sg.get("object")

        if sg_type == "deliver_box":
            risk = 2
            progress = 3
        elif sg_type == "avoid_deadlock":
            risk = 0
            progress = 2
        elif sg_type == "clear_path":
            risk = 1
            progress = 2
        elif sg_type == "reposition_box":
            risk = 1
            progress = 1
        else:
            risk = 1
            progress = 1

        return {
            "label": label,
            "description": f"{sg_type} for object {obj}",
            "heuristic_risk": risk,
            "heuristic_progress": progress,
            "derived_from_subgoal": sg,
        }

    option_a = sg_to_option("A", sg1)
    option_b = sg_to_option("B", sg2)

    return {
        "option_A": option_a,
        "option_B": option_b,
    }


def generate_horizon_choice_tasks(json_path: str, num_samples: int = 20):
    task_builder = LongShortHorizonChoiceTask()
    envs = load_envs_from_json(json_path)

    tasks = []
    sample_id = 0
    idx = 0

    while len(tasks) < num_samples and envs:
        env = envs[idx % len(envs)]
        options = _make_horizon_options_from_env(env)

        instance = task_builder.build(
            task_id=f"horizon_choice_{sample_id}",
            option_a=options["option_A"],
            option_b=options["option_B"],
            level_id=env.map.level_id,
            source="generated_planning_batch_v1",
        )
        tasks.append((task_builder, instance))
        sample_id += 1
        idx += 1

    return tasks


def generate_all_planning_tasks(
    json_path: str,
    num_samples_per_type: int = 20,
):
    all_tasks = {
        "candidate_subgoal_discovery": generate_subgoal_discovery_tasks(json_path, num_samples_per_type),
        "box_priority_ranking": generate_box_priority_tasks(json_path, num_samples_per_type),
        "box_target_assignment": generate_box_target_assignment_tasks(json_path, num_samples_per_type),
        "phase_recognition": generate_phase_recognition_tasks(json_path, num_samples_per_type),
        "subproblem_ordering": generate_subproblem_ordering_tasks(json_path, num_samples_per_type),
        "long_vs_short_horizon_choice": generate_horizon_choice_tasks(json_path, num_samples_per_type),
    }
    return all_tasks


if __name__ == "__main__":
    json_file_path = "custom_levels.json"

    all_tasks = generate_all_planning_tasks(
        json_path=json_file_path,
        num_samples_per_type=5,
    )

    for task_type, task_list in all_tasks.items():
        print(f"{task_type}: {len(task_list)}")

    for task_type, task_list in all_tasks.items():
        if task_list:
            task, instance = task_list[0]
            print(f"\n=== {task_type} sample ===")
            print(instance.task_id)
            print(json.dumps(instance.input_data, ensure_ascii=False, indent=2))