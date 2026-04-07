from typing import Callable, Dict, Tuple, Any

from llm_protocol.parser import parse_json_response
from llm_protocol.prompt_builder import (
    build_legality_prompt,
    build_transition_prompt,
    build_deadlock_detection_prompt,
    build_static_dead_squares_prompt,
    build_box_status_explanation_prompt,
    # build_subgoal_discovery_prompt,
    build_box_priority_prompt,
    build_box_target_assignment_prompt,
    # build_phase_recognition_prompt,
    # build_subproblem_ordering_prompt,
    # build_horizon_choice_prompt,
)


PromptBuilder = Callable[[Any], str]

TASK_PROMPT_REGISTRY: Dict[str, Tuple[str, PromptBuilder]] = {
    "action_legality": (
        "You are a precise Sokoban evaluator. Return JSON only.",
        build_legality_prompt,
    ),
    "state_transition": (
        "You are a precise Sokoban state simulator. Return JSON only.",
        build_transition_prompt,
    ),
    "deadlock_detection": (
        "You are a strict Sokoban deadlock analyzer. Return JSON only.",
        build_deadlock_detection_prompt,
    ),
    "static_dead_squares": (
        "You are a strict Sokoban structural analyzer. Return JSON only.",
        build_static_dead_squares_prompt,
    ),
    "box_status_explanation": (
        "You are a careful Sokoban state analyst. Return JSON only.",
        build_box_status_explanation_prompt,
    ),
    # "candidate_subgoal_discovery": (
    #     "You are a careful Sokoban planner. Return JSON only.",
    #     build_subgoal_discovery_prompt,
    # ),
    "box_priority_ranking": (
        "You are a careful Sokoban planner. Return JSON only.",
        build_box_priority_prompt,
    ),
    "box_target_assignment": (
        "You are a careful Sokoban planner. Return JSON only.",
        build_box_target_assignment_prompt,
    ),
    # "phase_recognition": (
    #     "You are a careful Sokoban planner. Return JSON only.",
    #     build_phase_recognition_prompt,
    # ),
    # "subproblem_ordering": (
    #     "You are a careful Sokoban planner. Return JSON only.",
    #     build_subproblem_ordering_prompt,
    # ),
    # "long_vs_short_horizon_choice": (
    #     "You are a careful Sokoban planner. Return JSON only.",
    #     build_horizon_choice_prompt,
    # ),
}

def get_prompt_config(task_type: str) -> Tuple[str, PromptBuilder]:
    if task_type not in TASK_PROMPT_REGISTRY:
        raise ValueError(f"Unsupported task type: {task_type}")
    return TASK_PROMPT_REGISTRY[task_type]

class GenericTaskModelAdapter:
    """
    统一模型适配器：
    - 根据 task_type 选择 system prompt 和 prompt builder
    - 调用底层 client.chat()
    - 统一 parse JSON
    """

    def __init__(self, client, model_name: str = "deepseek-chat"):
        self.client = client
        self.model_name = model_name

    def __call__(self, task_instance):
        system_prompt, prompt_builder = get_prompt_config(task_instance.task_type)
        user_prompt = prompt_builder(task_instance)

        raw_text = self.client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.model_name,
        )

        parsed_output, parse_error = parse_json_response(raw_text)
        if parse_error:
            return {
                "__parse_error__": parse_error,
                "__raw_text__": raw_text,
            }

        return parsed_output