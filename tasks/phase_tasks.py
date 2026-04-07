from typing import Dict, Any

from .base import BaseTask, TaskInstance, TaskResult
from core.state import SokobanMap, SokobanState
from core.serializer import serialize_full_state, render_text_map
from .utils.planning_ground_truth import infer_phase
from .utils.planning_case_utils import ALLOWED_PHASES


class PhaseRecognitionTask(BaseTask):
    task_type = "phase_recognition"

    def build(
        self,
        task_id: str,
        sokoban_map: SokobanMap,
        state: SokobanState,
        source: str = "manual",
    ) -> TaskInstance:
        gold_phase = infer_phase(sokoban_map, state)
        return TaskInstance(
            task_id=task_id,
            task_type=self.task_type,
            instruction="判断当前状态更像处于哪一种规划阶段，并给出简要理由。",
            input_data={
                "map_state": serialize_full_state(sokoban_map, state),
                "text_map": render_text_map(sokoban_map, state),
                "allowed_phases": sorted(ALLOWED_PHASES),
            },
            metadata={
                "level_id": sokoban_map.level_id,
                "source": source,
                "gold_phase": gold_phase,
            }
        )

    def evaluate(self, model_output: Dict[str, Any], task_instance: TaskInstance) -> TaskResult:
        if "__parse_error__" in model_output:
            return TaskResult(
                task_id=task_instance.task_id,
                task_type=task_instance.task_type,
                success=False,
                score=0.0,
                metrics={"parse_error": model_output["__parse_error__"]},
                feedback={"message": "Model output could not be parsed as valid JSON."},
                raw_output=model_output,
            )

        pred_phase = model_output.get("phase")
        valid = pred_phase in ALLOWED_PHASES
        gold_phase = task_instance.metadata["gold_phase"]

        success = valid and pred_phase == gold_phase
        score = 1.0 if success else (0.25 if valid else 0.0)

        return TaskResult(
            task_id=task_instance.task_id,
            task_type=task_instance.task_type,
            success=success,
            score=score,
            metrics={
                "predicted_phase": pred_phase,
                "gold_phase": gold_phase,
                "phase_valid": valid,
                "level_id": task_instance.metadata.get("level_id"),
                "source": task_instance.metadata.get("source"),
            },
            feedback={
                "message": "Correct phase recognition." if success else "Incorrect or unsupported phase label."
            },
            raw_output=model_output,
        )