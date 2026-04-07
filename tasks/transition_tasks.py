from typing import Dict, Any

from .base import BaseTask, TaskInstance, TaskResult
from core.env import SokobanEnv
from core.serializer import serialize_full_state, serialize_state
from core.comparator import compare_predicted_state


class StateTransitionTask(BaseTask):
    task_type = "state_transition"

    def build(
        self,
        task_id: str,
        env: SokobanEnv,
        action: str,
        case_type: str = "unknown",
    ) -> TaskInstance:
        return TaskInstance(
            task_id=task_id,
            task_type=self.task_type,
            instruction=(
                "给定当前 Sokoban 状态和一个动作，"
                "请预测执行该动作后的玩家位置与箱子位置。"
            ),
            input_data={
                "current_state": serialize_full_state(env.map, env.state),
                "action": action,
            },
            metadata={
                "env": env.clone(),
                "level_id": env.map.level_id,
                "case_type": case_type,
                "action": action,
                "num_boxes": len(env.state.boxes),
            }
        )

    def evaluate(self, model_output: Dict[str, Any], task_instance: TaskInstance) -> TaskResult:
        if "__parse_error__" in model_output:
            return TaskResult(
                task_id=task_instance.task_id,
                task_type=task_instance.task_type,
                success=False,
                score=0.0,
                metrics={
                    "parse_error": model_output["__parse_error__"]
                },
                feedback={
                    "message": "Model output could not be parsed as valid JSON."
                },
                raw_output=model_output,
            )
        env: SokobanEnv = task_instance.metadata["env"]
        action = task_instance.input_data["action"]

        step_result = env.step(action)
        actual_state = serialize_state(env.state)
        compare_result = compare_predicted_state(model_output, actual_state)

        score = 0.0
        if step_result.success:
            if compare_result.player_match:
                score += 0.5
            if compare_result.box_match:
                score += 0.5

        return TaskResult(
            task_id=task_instance.task_id,
            task_type=task_instance.task_type,
            success=compare_result.all_match and step_result.success,
            score=score,
            metrics={
                "action_success": step_result.success,
                "action_message": step_result.message,
                "player_match": compare_result.player_match,
                "box_match": compare_result.box_match,
                "error_type": compare_result.error_type,
                "actual_state": actual_state,
                "level_id": task_instance.metadata.get("level_id"),
                "case_type": task_instance.metadata.get("case_type"),
                "action": task_instance.metadata.get("action"),
                "num_boxes": task_instance.metadata.get("num_boxes"),
            },
            feedback={
                "message": compare_result.error_type
            },
            raw_output=model_output,
        )