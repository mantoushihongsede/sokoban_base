from typing import Dict, Any, List

from .base import BaseTask, TaskInstance, TaskResult
from .utils.planning_case_utils import pairwise_order_accuracy
from .utils.planning_ground_truth import order_subproblems


class SubproblemOrderingTask(BaseTask):
    task_type = "subproblem_ordering"

    def build(
        self,
        task_id: str,
        candidate_subgoals: List[Dict[str, Any]],
        level_id: str = "unknown",
        source: str = "manual",
    ) -> TaskInstance:
        gold_order = order_subproblems(candidate_subgoals)
        return TaskInstance(
            task_id=task_id,
            task_type=self.task_type,
            instruction="给定多个候选子任务，请判断更合理的执行顺序。",
            input_data={
                "candidate_subgoals": candidate_subgoals,
            },
            metadata={
                "level_id": level_id,
                "source": source,
                "gold_order": gold_order,
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

        pred_order = model_output.get("ordered_subgoals", [])
        if not isinstance(pred_order, list):
            pred_order = []

        gold_order = task_instance.metadata["gold_order"]
        pairwise = pairwise_order_accuracy(pred_order, gold_order)
        exact = 1.0 if pred_order == gold_order else 0.0
        coverage = len([x for x in pred_order if x in gold_order]) / max(1, len(gold_order))

        score = 0.6 * pairwise + 0.2 * exact + 0.2 * coverage
        success = pairwise >= 0.8

        return TaskResult(
            task_id=task_instance.task_id,
            task_type=task_instance.task_type,
            success=success,
            score=round(score, 4),
            metrics={
                "pairwise_order_score": round(pairwise, 4),
                "exact_match": exact,
                "coverage": round(coverage, 4),
                "gold_order": gold_order,
                "predicted_order": pred_order,
                "level_id": task_instance.metadata.get("level_id"),
                "source": task_instance.metadata.get("source"),
            },
            feedback={
                "message": "Ordering is largely consistent with dependency heuristic." if success else "Ordering conflicts with heuristic dependency structure."
            },
            raw_output=model_output,
        )