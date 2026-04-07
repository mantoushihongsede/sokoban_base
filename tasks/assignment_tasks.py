from typing import Dict, Any, List, Tuple

from .base import BaseTask, TaskInstance, TaskResult
from core.state import SokobanMap, SokobanState
from core.serializer import serialize_full_state, render_text_map
from tasks.box_priority_tasks import normalize_pos


class BoxTargetAssignmentTask(BaseTask):
    task_type = "box_target_assignment"

    def build(
        self,
        task_id: str,
        sokoban_map: SokobanMap,
        state: SokobanState,
        case_type: str = "box_target_assignment",
        gt: dict = None
    ) -> TaskInstance:
        gt = gt if gt is not None else {}

        return TaskInstance(
            task_id=task_id,
            task_type=self.task_type,
            instruction="分析当前状态，将箱子与更合适的目标点进行匹配。",
            input_data={
                "map_state": serialize_full_state(sokoban_map, state),
                "text_map": render_text_map(sokoban_map, state),
            },
            metadata={
                "case_type": case_type,
                "level_id": sokoban_map.level_id,
                "ground_truth": gt,
            }
        )

    def evaluate(self, model_output: Dict[str, Any], task_instance: TaskInstance) -> TaskResult:
        if "__parse_error__" in model_output:
            feedback = {
                "parse_ok": False,
                "format_ok": False,
                "message": "Model output could not be parsed as valid JSON.",
                "parse_error": model_output["__parse_error__"],
                "expected_format": {
                    "assignments": [
                        {"box": "[row, col]", "target": "[row, col]"}
                    ]
                },
            }
            metrics = {
                "exact_match": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "count_match": 0.0,
            }

            return TaskResult(
                task_id=task_instance.task_id,
                task_type=task_instance.task_type,
                success=False,
                score=0.0,
                metrics=metrics,
                feedback=feedback,
                raw_output=model_output,
            )

        gt_raw = task_instance.metadata.get("ground_truth", [])
        gt_assignments = []
        for item in gt_raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                box = normalize_pos(item[0])
                target = normalize_pos(item[1])
                if box is not None and target is not None:
                    gt_assignments.append((box, target))

        pred_raw = model_output.get("assignments", None)

        format_ok = isinstance(pred_raw, list)
        invalid_items = []
        pred_assignments = []

        if isinstance(pred_raw, list):
            for i, item in enumerate(pred_raw):
                if not isinstance(item, dict):
                    invalid_items.append({"index": i, "value": item})
                    continue

                box = normalize_pos(item.get("box"))
                target = normalize_pos(item.get("target"))

                if box is None or target is None:
                    invalid_items.append({"index": i, "value": item})
                    continue

                pred_assignments.append((box, target))
        else:
            pred_assignments = []

        gt_count = len(gt_assignments)
        pred_count = len(pred_assignments)

        gt_set = set(gt_assignments)
        pred_set = set(pred_assignments)

        matched_set = gt_set & pred_set
        missing_set = gt_set - pred_set
        extra_set = pred_set - gt_set

        exact_match = 1.0 if pred_assignments == gt_assignments else 0.0
        count_match = 1.0 if pred_count == gt_count else 0.0

        precision = len(matched_set) / max(1, pred_count)
        recall = len(matched_set) / max(1, gt_count)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)

        success = bool(exact_match == 1.0)
        score = 1.0 if exact_match == 1.0 else round(f1, 4)

        feedback = {
            "parse_ok": True,
            "format_ok": format_ok,
            "message": (
                "Predicted assignments exactly match ground truth."
                if success else
                "Predicted assignments do not exactly match ground truth."
            ),
            "expected_count": gt_count,
            "predicted_count": pred_count,
            "expected_assignments": [
                {"box": list(box), "target": list(target)} for box, target in gt_assignments
            ],
            "predicted_assignments": [
                {"box": list(box), "target": list(target)} for box, target in pred_assignments
            ],
            "matched_assignments": [
                {"box": list(box), "target": list(target)} for box, target in sorted(matched_set)
            ],
            "missing_assignments": [
                {"box": list(box), "target": list(target)} for box, target in sorted(missing_set)
            ],
            "extra_assignments": [
                {"box": list(box), "target": list(target)} for box, target in sorted(extra_set)
            ],
            "invalid_items": invalid_items,
        }

        metrics = {
            "exact_match": exact_match,
            "count_match": count_match,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

        return TaskResult(
            task_id=task_instance.task_id,
            task_type=task_instance.task_type,
            success=success,
            score=score,
            metrics=metrics,
            feedback=feedback,
            raw_output=model_output,
        )