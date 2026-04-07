from typing import Dict, Any

from .base import BaseTask, TaskInstance, TaskResult
from .utils.planning_ground_truth import choose_better_horizon_option


class LongShortHorizonChoiceTask(BaseTask):
    task_type = "long_vs_short_horizon_choice"

    def build(
        self,
        task_id: str,
        option_a: Dict[str, Any],
        option_b: Dict[str, Any],
        level_id: str = "unknown",
        source: str = "manual",
    ) -> TaskInstance:
        gold_choice = choose_better_horizon_option(option_a, option_b)
        return TaskInstance(
            task_id=task_id,
            task_type=self.task_type,
            instruction=(
                "在两个候选方案中选择更优者。"
                "一个方案可能短期更直接，另一个方案可能长期更安全。"
            ),
            input_data={
                "option_A": option_a,
                "option_B": option_b,
            },
            metadata={
                "level_id": level_id,
                "source": source,
                "gold_choice": gold_choice,
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

        pred_choice = model_output.get("better_choice")
        gold_choice = task_instance.metadata["gold_choice"]

        valid = pred_choice in {"A", "B"}
        success = valid and pred_choice == gold_choice
        score = 1.0 if success else (0.2 if valid else 0.0)

        return TaskResult(
            task_id=task_instance.task_id,
            task_type=task_instance.task_type,
            success=success,
            score=score,
            metrics={
                "predicted_choice": pred_choice,
                "gold_choice": gold_choice,
                "choice_valid": valid,
                "level_id": task_instance.metadata.get("level_id"),
                "source": task_instance.metadata.get("source"),
            },
            feedback={
                "message": "Correct horizon-aware choice." if success else "Incorrect long-vs-short horizon choice."
            },
            raw_output=model_output,
        )