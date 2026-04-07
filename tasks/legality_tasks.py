from typing import Dict, Any

from .base import BaseTask, TaskInstance, TaskResult
from core.state import SokobanMap, SokobanState
from core.env import SokobanEnv
from core.serializer import serialize_full_state


class ActionLegalityTask(BaseTask):
    task_type = "action_legality"

    def build(
        self,
        task_id: str,
        sokoban_map: SokobanMap,
        state: SokobanState,
        candidate_action: str,
        case_type: str = "unknown",
        source: str = "manual",
    ) -> TaskInstance:
        env = SokobanEnv(sokoban_map, state)
        return TaskInstance(
            task_id=task_id,
            task_type=self.task_type,
            instruction="判断给定动作是否合法，并说明原因。",
            input_data={
                "map_state": serialize_full_state(sokoban_map, state),
                "candidate_action": candidate_action,
            },
            metadata={
            "env": env.clone(),
            "level_id": sokoban_map.level_id,
            "case_type": case_type,
            "candidate_action": candidate_action,
            "num_boxes": len(state.boxes),
            "source": source,
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
        
        # input_data = task_instance.input_data
        # candidate_action = input_data["candidate_action"]
        # map_state = input_data["map_state"]

        # 这里实际用时最好传原始对象，不只传序列化
        # 当前先假设 metadata 里有 env
        env: SokobanEnv = task_instance.metadata["env"]
        candidate_action = task_instance.input_data["candidate_action"]

        legal = env.is_legal_move(candidate_action)
        predicted_legal = model_output.get("legal", None)

        success = (predicted_legal == legal)
        score = 1.0 if success else 0.0

        return TaskResult(
            task_id=task_instance.task_id,
            task_type=task_instance.task_type,
            success=success,
            score=score,
            metrics={
                "ground_truth_legal": legal,
                "predicted_legal": predicted_legal,
                "level_id": task_instance.metadata.get("level_id"),
                "case_type": task_instance.metadata.get("case_type"),
                "candidate_action": task_instance.metadata.get("candidate_action"),
                "num_boxes": task_instance.metadata.get("num_boxes"),
                "source": task_instance.metadata.get("source"),
            },
            feedback={
                "message": "Correct legality judgment." if success else "Incorrect legality judgment."
            },
            raw_output=model_output,
        )