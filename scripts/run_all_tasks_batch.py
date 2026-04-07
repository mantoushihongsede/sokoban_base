import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Callable, List, Tuple

import concurrent.futures
from dotenv import dotenv_values
from tqdm import tqdm 
from threading import RLock


# from dotenv import load_dotenv

from evaluation.runner import EpisodeRunner
from llm_protocol.client_factory import create_client
from llm_protocol.task_inference import GenericTaskModelAdapter

from tasks.utils.generate_batch_tasks import (
    generate_legality_tasks,
    generate_transition_tasks,
)

from tasks.utils.generate_deadlock_tasks import (
    generate_definite_deadlock_tasks, 
    generate_static_dead_squares_tasks, 
    generate_box_status_explanation_tasks,
)

from tasks.utils.generate_order_match_tasks import (
    generate_box_priority_tasks,
    generate_box_target_assignment_tasks,
)

from tasks.utils.summarize_results import (
    summarize_results,
    summarize_transition_results,
    summarize_deadlock_detection_results,
    summarize_static_dead_results,
    summarize_box_explanation_results,
    summarize_subgoal_results,
    summarize_box_priority_results,
    summarize_assignment_results,
    summarize_phase_results,
    summarize_subproblem_ordering_results,
    summarize_horizon_choice_results,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =========================
# 输出工具
# =========================

def print_some_cases(task_results, limit=5):
    print("\n=== Sample Cases ===")
    for result in task_results[:limit]:
        print(json.dumps({
            "task_id": result.task_id,
            "task_type": result.task_type,
            "success": result.success,
            "score": result.score,
            "metrics": result.metrics,
            "feedback": result.feedback,
            "raw_output": result.raw_output,
        }, ensure_ascii=False, indent=2))
        print("-" * 40)


def save_task_results_to_txt(task_results, file_path, title="Task Results"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write("=" * 80 + "\n\n")

        for i, result in enumerate(task_results, start=1):
            record = {
                "index": i,
                "task_id": result.task_id,
                "task_type": result.task_type,
                "success": result.success,
                "score": result.score,
                "metrics": result.metrics,
                "feedback": result.feedback,
                "raw_output": result.raw_output,
            }
            f.write(json.dumps(record, ensure_ascii=False, indent=2))
            f.write("\n" + "-" * 80 + "\n\n")


def save_summary_to_txt(summary, file_path, title="Summary"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write("=" * 80 + "\n\n")
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))
        f.write("\n")


# =========================
# 注册式管理：任务生成器
# =========================

TASK_GENERATOR_REGISTRY: Dict[str, Callable[..., List[Tuple[Any, Any]]]] = {
    "action_legality": generate_legality_tasks,
    "state_transition": generate_transition_tasks,
    "deadlock_detection": generate_definite_deadlock_tasks,
    "static_dead_squares": generate_static_dead_squares_tasks,
    "box_status_explanation": generate_box_status_explanation_tasks,
    # "candidate_subgoal_discovery": generate_subgoal_discovery_tasks,
    "box_priority_ranking": generate_box_priority_tasks,
    "box_target_assignment": generate_box_target_assignment_tasks,
    # "phase_recognition": generate_phase_recognition_tasks,
    # "subproblem_ordering": generate_subproblem_ordering_tasks,
    # "long_vs_short_horizon_choice": generate_horizon_choice_tasks,
}


# =========================
# 注册式管理：summary 函数
# =========================

TASK_SUMMARY_REGISTRY: Dict[str, Callable[[List[Any]], Dict[str, Any]]] = {
    "action_legality": summarize_results,
    "state_transition": summarize_transition_results,
    "deadlock_detection": summarize_deadlock_detection_results,
    "static_dead_squares": summarize_static_dead_results,
    "box_status_explanation": summarize_box_explanation_results,
    # "candidate_subgoal_discovery": summarize_subgoal_results,
    "box_priority_ranking": summarize_box_priority_results,
    "box_target_assignment": summarize_assignment_results,
#     "phase_recognition": summarize_phase_results,
#     "subproblem_ordering": summarize_subproblem_ordering_results,
#     "long_vs_short_horizon_choice": summarize_horizon_choice_results,
}


# =========================
# 注册式管理：展示名称
# =========================

TASK_TITLE_REGISTRY: Dict[str, str] = {
    "action_legality": "Action Legality",
    "state_transition": "State Transition",
    "deadlock_detection": "Definite Deadlock",
    "static_dead_squares": "Static Dead Squares",
    "box_status_explanation": "Box Status Explanation",
    # "candidate_subgoal_discovery": "Candidate Subgoal Discovery",
    "box_priority_ranking": "Box Priority Ranking",
    "box_target_assignment": "Box-Target Assignment",
    # "phase_recognition": "Phase Recognition",
    # "subproblem_ordering": "Subproblem Ordering",
    # "long_vs_short_horizon_choice": "Long-vs-Short Horizon Choice",
}


def generate_tasks_by_type(task_type: str, json_path: str, num_samples: int):
    if task_type not in TASK_GENERATOR_REGISTRY:
        raise ValueError(f"Unsupported task type for generation: {task_type}")

    generator = TASK_GENERATOR_REGISTRY[task_type]

    if task_type == "state_transition":
        return generator(json_path=json_path, num_samples=num_samples, legal_only=True)

    return generator(json_path=json_path, num_samples=num_samples)


def summarize_by_task_type(task_type: str, task_results):
    if task_type not in TASK_SUMMARY_REGISTRY:
        return summarize_results(task_results)
    return TASK_SUMMARY_REGISTRY[task_type](task_results)


def run_one_group(
    runner,
    task_type: str,
    task_list,
    output_dir: str,
    timestamp: str,
):
    title = TASK_TITLE_REGISTRY.get(task_type, task_type)

    record = runner.run_tasks(
        episode_id=f"batch_{task_type}_exp",
        task_and_instances=task_list,
    )
    summary = summarize_by_task_type(task_type, record.task_results)

    # print(f"\n=== {title} Summary ===")
    # print(json.dumps(summary, ensure_ascii=False, indent=2))
    # print_some_cases(record.task_results, limit=3)

    save_summary_to_txt(
        summary,
        os.path.join(output_dir, f"{task_type}_summary_{timestamp}.txt"),
        title=f"{title} Summary",
    )
    save_task_results_to_txt(
        record.task_results,
        os.path.join(output_dir, f"{task_type}_task_results_{timestamp}.txt"),
        title=f"{title} Task Results",
    )

def run_single_model(
    provider: str, 
    model_name: str, 
    selected_task_types: list, 
    json_path: str, 
    num_samples: int, 
    timestamp: str
):
    env_path = PROJECT_ROOT / "env" / f".env.{provider}"
    env_config = dotenv_values(env_path)
    
    api_key = env_config.get("API_KEY")
    url = env_config.get("BASE_URL")

    if not api_key:
        return f"[Error] API_KEY not found for {provider}"

    # 初始化客户端与运行器
    client = create_client(provider=provider, api_key=api_key, base_url=url)
    adapter = GenericTaskModelAdapter(client, model_name=model_name)
    runner = EpisodeRunner(adapter)

    output_dir = os.path.join(PROJECT_ROOT, "outputs", provider, model_name, "batch")
    os.makedirs(output_dir, exist_ok=True)

    # 移除 tqdm，直接遍历任务类型
    for task_type in selected_task_types:
        task_list = generate_tasks_by_type(
            task_type=task_type,
            json_path=json_path,
            num_samples=num_samples,
        )

        if task_list:
            run_one_group(
                runner=runner,
                task_type=task_type,
                task_list=task_list,
                output_dir=output_dir,
                timestamp=timestamp,
            )
            # 如果你想在控制台看到当前执行到的具体任务类型，可以取消下面这行的注释
            # print(f"[{provider} - {model_name}] Finished task: {task_type}")
    
    return f"[Finished] {provider} - {model_name}"


def main():
    json_path = str(PROJECT_ROOT / "datasets" / "test_1.json")
    num_samples_per_type = 5
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 配置：字典形式，支持一个供应商对应多个模型
    # 下面的快慢都是不限制思考深度，模型情况下的速度。
    # 实际测试时，尤其是Gemini，可能需要调整thinking_level参数来控制速度和成本。可以去gemini_client里根据模型名称做特殊处理。
    models_to_run = {
        "openai": [
            # "gpt-5.4-mini", # 0.75/4.5 快
            # "gpt-5.4-nano", # 0.2/1.25 快
            # "gpt-5.4", # 2.5/15 快
            # "gpt-5.2", # 1.75/14 快
            # "gpt-5.1", # 1.25/10 快
        #     "gpt-5-mini", # 0.25/2 慢 192.30s
        #     "gpt-5-nano", # 0.05/0.4 超级慢 351.38s/it
        #     # "gpt-5", # 1.25/10 超级慢 445.45s/it
        ],
        "gemini": [
            # "gemini-3.1-pro-preview", # 2/12 慢 246.76s
            # "gemini-3.1-flash-lite-preview", #0.25/1.5
        #     # "gemini-3-flash-preview", # 0.5/3 超级慢 
        #     # "gemini-2.5-flash", # 0.3/2.5 超级慢 358.66s/it
        #     # "gemini-2.5-pro", # 1.25/10 慢 293.55s/it
        ],
        "claude": [
            # "claude-haiku-4-5", # 1/5 快
        #     # "claude-sonnet-4-5", # 3/15 可以接受 65.05s/it 要单独测试，并去claude_client跟据测试任务单独修改transform_schema的输入
        #     # "claude-sonnet-4-0" # 3/15 39.38s/it 同上
        ],
        "deepseek": [
            "deepseek-chat", # 默认正式跑数优先使用 DeepSeek 官方接口
            # "deepseek-reasoner" # 2/3 RMB 超级慢
        ],
    }

    selected_task_types = [
        "action_legality",
        "state_transition",
        "deadlock_detection", #1
        "static_dead_squares", #1
        "box_status_explanation", #1

        # "box_priority_ranking", #2
        # "box_target_assignment", #2

        # "candidate_subgoal_discovery",
        # "phase_recognition",
        # "subproblem_ordering",
        # "long_vs_short_horizon_choice",
    ]

    tasks = []
    for provider, model_list in models_to_run.items():
        for model_name in model_list:
            tasks.append((provider, model_name))

    print(f"Starting {len(tasks)} model evaluation tasks concurrently...\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for provider, model_name in tasks:
            future = executor.submit(
                run_single_model, 
                provider, 
                model_name, 
                selected_task_types, 
                json_path, 
                num_samples_per_type, 
                timestamp
                # 已移除 idx 参数
            )
            futures[future] = (provider, model_name)

        for future in concurrent.futures.as_completed(futures):
            provider, model_name = futures[future]
            try:
                future.result() 
                # 增加标准打印，替代进度条的视觉反馈
                print(f"[Completed] {provider} - {model_name}")
            except Exception as e:
                # 使用 print 替代 tqdm.write
                print(f"[Exception] {provider} - {model_name}: {e}")

    print("\nAll concurrent evaluations have finished.")

if __name__ == "__main__":
    main()


#当测试claude时，推荐用这里main。因为要跟据测试的任务单独调整transform_schema的输入，
#需要跟据任务要求去claude_client里调整对应的schema。

# def main():
#     # provider = "deepseek"   # deepseek / openai / gemini / claude
#     # model_name = "deepseek-chat"  # 替换为你想使用的模型名称deepseek-reasoner

#     # provider = "openai"   # deepseek / openai / gemini / claude
#     # model_name = "gpt-5.4-mini-2026-03-17"  # 替换为你想使用的模型名称deepseek-reasoner

#     provider = "gemini"   # deepseek / openai / gemini / claude
#     model_name = "gemini-3-flash-preview"  # 替换为你想使用的模型名称deepseek-reasoner

#     # provider = "claude"   # deepseek / openai / gemini / claude
#     # model_name = "claude-haiku-4-5-20251001"


#     # json_path = r"F:\code_repositories\sokoban\datasets\test_1.json"
#     json_path = r"F:\code_repositories\sokoban\datasets\test_1.json"
#     num_samples_per_type = 5

#     # 你可以在这里自由选择要跑哪些任务
#     selected_task_types = [
#         # "action_legality",
#         "state_transition",
#         # "deadlock_detection", #1
#         # "static_dead_squares", #1
#         # "box_status_explanation", #1
#         # "candidate_subgoal_discovery",
#         # "box_priority_ranking", #2
#         # "box_target_assignment", #2
#         # "phase_recognition",
#         # "subproblem_ordering",
#         # "long_vs_short_horizon_choice",
#     ]

#     load_dotenv(f"env\\.env.{provider}", override=True)
#     api_key = os.getenv("API_KEY")
#     url = os.getenv("BASE_URL")
#     client = create_client(provider=provider, api_key=api_key, base_url=url)

#     adapter = GenericTaskModelAdapter(client, model_name=model_name)
#     runner = EpisodeRunner(adapter)

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_dir = os.path.join("outputs", provider, model_name, "batch")
#     os.makedirs(output_dir, exist_ok=True)

#     for task_type in selected_task_types:
#         print(f"\n[Running] {task_type}")
#         task_list = generate_tasks_by_type(
#             task_type=task_type,
#             json_path=json_path,
#             num_samples=num_samples_per_type,
#         )

#         if not task_list:
#             print(f"[Skip] No tasks generated for {task_type}")
#             continue

#         run_one_group(
#             runner=runner,
#             task_type=task_type,
#             task_list=task_list,
#             output_dir=output_dir,
#             timestamp=timestamp,
#         )

#     print(f"\nAll results have been saved to: {output_dir}")
