from typing import Dict, Any, List, Tuple, Optional

from .base import BaseTask, TaskInstance, TaskResult
from core.state import SokobanMap, SokobanState, Pos
from core.serializer import serialize_full_state, render_text_map

def normalize_pos(pos: Any) -> Optional[Pos]:
    if not isinstance(pos, (list, tuple)) or len(pos) != 2:
        return None
    if not all(isinstance(x, int) for x in pos):
        return None
    return (pos[0], pos[1])


class BoxPriorityRankingTask(BaseTask):
    task_type = "box_priority_ranking"

    def build(
        self,
        task_id: str,
        sokoban_map: SokobanMap,
        state: SokobanState,
        case_type: str = "box_priority",
        gt: dict = None
    ) -> TaskInstance:
        gt = gt if gt is not None else {}

        return TaskInstance(
            task_id=task_id,
            task_type=self.task_type,
            instruction="分析当前状态，判断哪些箱子应该优先处理，并给出排序。",
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
                    "box_priority_order": "[[row, col], [row, col]]"
                },
            }
            metrics = {
                "exact_match": 0.0,
                "top1_accuracy": 0.0,
                "prefix_match_ratio": 0.0,
                "pairwise_order_score": 0.0,
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
        gt_order = []
        for item in gt_raw:
            pos = normalize_pos(item)
            if pos is not None:
                gt_order.append(pos)

        pred_raw = model_output.get("box_priority_order", None)

        format_ok = isinstance(pred_raw, list)
        invalid_items = []
        pred_order = []

        if isinstance(pred_raw, list):
            for i, item in enumerate(pred_raw):
                pos = normalize_pos(item)
                if pos is None:
                    invalid_items.append({"index": i, "value": item})
                else:
                    pred_order.append(pos)
        else:
            pred_order = []

        gt_count = len(gt_order)
        pred_count = len(pred_order)

        top1_accuracy = 1.0 if gt_order and pred_order and gt_order[0] == pred_order[0] else 0.0

        exact_match = 1.0 if pred_order == gt_order else 0.0

        prefix_match_count = 0
        for gt_pos, pred_pos in zip(gt_order, pred_order):
            if gt_pos == pred_pos:
                prefix_match_count += 1
            else:
                break
        prefix_match_ratio = prefix_match_count / max(1, gt_count)

        gt_index = {pos: i for i, pos in enumerate(gt_order)}
        pred_index = {pos: i for i, pos in enumerate(pred_order)}
        common_positions = [pos for pos in gt_order if pos in pred_index]

        pair_total = 0
        pair_correct = 0
        for i in range(len(common_positions)):
            for j in range(i + 1, len(common_positions)):
                a = common_positions[i]
                b = common_positions[j]
                pair_total += 1
                if pred_index[a] < pred_index[b]:
                    pair_correct += 1

        pairwise_order_score = pair_correct / pair_total if pair_total > 0 else 0.0

        missing_positions = [list(pos) for pos in gt_order if pos not in pred_order]
        extra_positions = [list(pos) for pos in pred_order if pos not in gt_order]

        success = bool(exact_match == 1.0)
        score = 1.0 if exact_match == 1.0 else round(
            0.5 * top1_accuracy + 0.3 * prefix_match_ratio + 0.2 * pairwise_order_score, 4
        )

        feedback = {
            "parse_ok": True,
            "format_ok": format_ok,
            "message": (
                "Predicted box priority order exactly matches ground truth."
                if success else
                "Predicted box priority order does not exactly match ground truth."
            ),
            "expected_count": gt_count,
            "predicted_count": pred_count,
            "expected_box_priority_order": [list(pos) for pos in gt_order],
            "predicted_box_priority_order": [list(pos) for pos in pred_order],
            "top1_expected": list(gt_order[0]) if gt_order else None,
            "top1_predicted": list(pred_order[0]) if pred_order else None,
            "matching_prefix_length": prefix_match_count,
            "missing_positions": missing_positions,
            "extra_positions": extra_positions,
            "invalid_items": invalid_items,
        }

        metrics = {
            "exact_match": exact_match,
            "top1_accuracy": top1_accuracy,
            "prefix_match_ratio": round(prefix_match_ratio, 4),
            "pairwise_order_score": round(pairwise_order_score, 4),
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