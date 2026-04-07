from typing import Any, Dict, List, Optional, Tuple, Set

from core.state import SokobanMap, SokobanState, Pos
from core.serializer import serialize_full_state
from tasks.base import BaseTask, TaskInstance, TaskResult
from analyzers.tools import SokobanAnalysisTools


def _normalize_pos_list(items: List[List[int]]) -> List[List[int]]:
    return sorted([[int(p[0]), int(p[1])] for p in items])


def _normalize_reason_list(items: List[str]) -> List[str]:
    return sorted([str(x).strip() for x in items if str(x).strip()])


def _safe_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    return None


def _normalize_reason(reason: Any) -> str:
    return str(reason).strip().lower()

def _parse_box_reason_mapping(items: Any) -> Dict[Pos, Set[str]]:
    """
    支持两种格式：

    格式A：按箱子聚合
    [
        [[1, 1], ["static dead square", "blocked 2x2 structure"]],
        [[2, 2], ["static dead square"]],
    ]

    格式B：扁平 pair
    [
        [[1, 1], "static dead square"],
        [[1, 1], "blocked 2x2 structure"],
        [[2, 2], "static dead square"],
    ]
    """
    result: Dict[Pos, Set[str]] = {}

    if not isinstance(items, list):
        return result

    for item in items:
        if not isinstance(item, list) or len(item) != 2:
            continue

        pos_raw, reasons_raw = item

        if not (isinstance(pos_raw, list) and len(pos_raw) == 2):
            continue

        try:
            pos = (int(pos_raw[0]), int(pos_raw[1]))
        except Exception:
            continue

        if pos not in result:
            result[pos] = set()

        # 情况1：reasons_raw 是一个字符串
        if isinstance(reasons_raw, str):
            reason = _normalize_reason(reasons_raw)
            if reason:
                result[pos].add(reason)

        # 情况2：reasons_raw 是一个列表
        elif isinstance(reasons_raw, list):
            for r in reasons_raw:
                reason = _normalize_reason(r)
                if reason:
                    result[pos].add(reason)

    return result


class DefiniteDeadlockDetectionTask(BaseTask):
    """
    任务目标：
    给定 Sokoban 当前状态，判断它是否已经处于“严格死锁”状态。

    模型输出格式：
    {
        "is_deadlock": true / false,
        "dead_positions": [[[row, col], definite_reasons],...]           # optional
    }
    """

    task_type = "deadlock_detection"

    def __init__(self) -> None:
        self.tools = SokobanAnalysisTools()

    def build(
        self,
        task_id: str,
        sokoban_map: SokobanMap,
        state: SokobanState,
        case_type: str = "definite_deadlock_detection",
        gt: dict = None
    ) -> TaskInstance:
        gt = gt if gt is not None else {}

        input_data = {
            "state": serialize_full_state(sokoban_map, state),
            "expected_output_schema": {
                "is_deadlock": "bool",
                "definite_reasons": "[[[row, col], [reason_tag, ...]], ...]",
            },
        }

        metadata = {
            "case_type": case_type,
            "ground_truth": gt,
            "level_id": sokoban_map.level_id,
        }
        # 这里并不会被传给大模型，而且尚未更新，仅作保留字段。提示词在prompt_builder里。
        instruction = (
            "You are given a Sokoban state. "
            "Judge whether the state is definitely deadlocked using only strict deadlock rules. "
            "Return JSON with field 'is_deadlock'. "
            "If deadlocked, also provide 'definite_reasons' in the format "
            "[[[row, col], [reason_tag, ...]], ...], where each entry corresponds to one deadlocked box."
)

        return TaskInstance(
            task_id=task_id,
            task_type=self.task_type,
            instruction=instruction,
            input_data=input_data,
            metadata=metadata,
        )

    def evaluate(self, model_output: Dict[str, Any], task_instance: TaskInstance) -> TaskResult:
        gt = task_instance.metadata.get("ground_truth", {}) or {}

        pred_is_deadlock = _safe_bool(model_output.get("is_deadlock"))
        gt_is_deadlock = bool(gt.get("is_deadlock", False))

        is_deadlock_correct = (
            pred_is_deadlock is not None and pred_is_deadlock == gt_is_deadlock
        )

        # 统一解析 GT 和模型输出中的 definite_reasons
        gt_box_reasons = _parse_box_reason_mapping(gt.get("definite_reasons", []))
        pred_box_reasons = _parse_box_reason_mapping(model_output.get("definite_reasons", []))

        gt_boxes = set(gt_box_reasons.keys())

        strict_correct_boxes = 0
        partial_correct_boxes = 0

        for box_pos, gt_reasons in gt_box_reasons.items():
            pred_reasons = pred_box_reasons.get(box_pos, set())

            # 严格：该箱子的 reasons 完全一致
            if pred_reasons == gt_reasons and len(gt_reasons) > 0:
                strict_correct_boxes += 1

            # 宽松：该箱子只要命中任意一个 GT reason 就算对
            if len(pred_reasons & gt_reasons) > 0:
                partial_correct_boxes += 1

        total_gt_boxes = len(gt_boxes)

        if total_gt_boxes > 0:
            strict_box_reason_accuracy = strict_correct_boxes / total_gt_boxes
            partial_box_reason_accuracy = partial_correct_boxes / total_gt_boxes
        else:
            # 没有 deadlock boxes 时的处理
            strict_box_reason_accuracy = 1.0 if not pred_box_reasons else 0.0
            partial_box_reason_accuracy = 1.0 if not pred_box_reasons else 0.0

        # success 仍然主要由 is_deadlock 判断决定
        success = is_deadlock_correct

        # score 你可以按你的需求组合
        # 如果 is_deadlock 判断错误，直接 0 分；如果正确，根据 box reason 的部分正确率给分。
        if not is_deadlock_correct:
            score = 0.0
        else:
            if gt_is_deadlock:
                score = 0.6 + 0.4 * partial_box_reason_accuracy
            else:
                score = 1.0

        feedback = {
            "expected_is_deadlock": gt_is_deadlock,
            "predicted_is_deadlock": pred_is_deadlock,
            "correct_is_deadlock": is_deadlock_correct,
            "expected_definite_reasons": {
                str(k): sorted(list(v)) for k, v in gt_box_reasons.items()
            },
            "predicted_definite_reasons": {
                str(k): sorted(list(v)) for k, v in pred_box_reasons.items()
            },
            "strict_correct_boxes": strict_correct_boxes,
            "partial_correct_boxes": partial_correct_boxes,
            "total_gt_boxes": total_gt_boxes,
        }

        metrics = {
            "is_deadlock_accuracy": 1.0 if is_deadlock_correct else 0.0,
            "strict_box_reason_accuracy": strict_box_reason_accuracy,
            "partial_box_reason_accuracy": partial_box_reason_accuracy,
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


class StaticDeadSquaresTask(BaseTask):
    """
    任务目标：
    给定地图和当前状态，让模型识别地图中的所有 static dead squares。

    这里虽然只和地图有关，但为了保持和现有任务输入风格一致，
    仍然把 full_state 一并给模型，方便它理解关卡上下文。
    """

    task_type = "static_dead_squares"

    def __init__(self) -> None:
        self.tools = SokobanAnalysisTools()

    def build(
        self,
        task_id: str,
        sokoban_map: SokobanMap,
        state: SokobanState,
        case_type: str = "static_dead_squares",
        gt: dict = None
    ) -> TaskInstance:
        gt = gt if gt is not None else {}

        input_data = {
            "state": serialize_full_state(sokoban_map, state),
            "expected_output_schema": {
                "static_dead_squares": "[[row, col], ...]",
            },
        }

        metadata = {
            "case_type": case_type,
            "ground_truth": gt,
            "level_id": sokoban_map.level_id,
        }

        instruction = (
            "You are given a Sokoban map and state. "
            "Identify all static dead squares and return them as JSON field 'static_dead_squares'."
        )

        return TaskInstance(
            task_id=task_id,
            task_type=self.task_type,
            instruction=instruction,
            input_data=input_data,
            metadata=metadata,
        )

    def evaluate(self, model_output: Dict[str, Any], task_instance: TaskInstance) -> TaskResult:
        gt = task_instance.metadata.get("ground_truth", {}) or {}

        gt_positions = _normalize_pos_list(gt.get("static_dead_squares", []))
        gt_count = gt.get("count", len(gt_positions))

        pred_positions_raw = model_output.get("static_dead_squares", [])
        pred_positions = []
        parse_ok = True
        if isinstance(pred_positions_raw, list):
            try:
                pred_positions = _normalize_pos_list(pred_positions_raw)
            except Exception:
                pred_positions = []
                parse_ok = False
        else:
            parse_ok = False

        gt_set = {tuple(x) for x in gt_positions}
        pred_set = {tuple(x) for x in pred_positions}
        pred_count = len(pred_set)

        exact_match = gt_set == pred_set
        count_match = (pred_count == gt_count)
        count_diff = abs(pred_count - gt_count)

        if not gt_set and not pred_set:
            precision, recall, f1 = 1.0, 1.0, 1.0
        else:
            precision = len(gt_set & pred_set) / pred_count if pred_count > 0 else 0.0
            recall = len(gt_set & pred_set) / len(gt_set) if gt_set else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        feedback = {
            "parse_ok": parse_ok,
            "expected_count": gt_count,
            "predicted_count": pred_count,
            "expected_static_dead_squares": gt_positions,
            "predicted_static_dead_squares": pred_positions,
            "missing_positions": sorted([list(p) for p in (gt_set - pred_set)]),
            "extra_positions": sorted([list(p) for p in (pred_set - gt_set)]),
        }

        metrics = {
            "exact_match": 1.0 if exact_match else 0.0,
            "count_match": 1.0 if count_match else 0.0,
            "count_diff": count_diff,  # 数量的绝对误差
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        return TaskResult(
            task_id=task_instance.task_id,
            task_type=task_instance.task_type,
            success=exact_match,
            score=1.0 if exact_match else f1,
            metrics=metrics,
            feedback=feedback,
            raw_output=model_output,
        )


class BoxStatusExplanationTask(BaseTask):
    """
    任务目标：
    给定一个指定箱子，解释其当前状态。

    模型输出建议格式：
    {
        "on_target": bool,
        "on_static_dead_square": bool,
        "in_blocked_2x2": bool,
        "warning_messages": ["...", ...]
    }
    """

    task_type = "box_status_explanation"

    def __init__(self) -> None:
        self.tools = SokobanAnalysisTools()

    def build(
        self,
        task_id: str,
        sokoban_map: SokobanMap,
        state: SokobanState,
        case_type: str = "box_status_explanation",
        gt: list = None,
    ) -> TaskInstance:
        gt = gt if gt is not None else []

        input_data = {
            "state": serialize_full_state(sokoban_map, state),
            "expected_output_schema": {
                "boxes": [
                    {
                        "box": "[row, col]",#必须得加个引号，别别扭扭。这个字段要不删了？
                        "on_target": "bool",
                        "on_static_dead_square": "bool",
                        "in_blocked_2x2": "bool",
                        "currently_immovable": "bool",
                        "legal_pushes": "[str, ...]",
                    }
                ]
            },
        }

        metadata = {
            "case_type": case_type,
            "ground_truth": gt,
            "level_id": sokoban_map.level_id,
        }

        instruction = (
            "You are given a Sokoban state. "
            "Classify all boxes in the state and return the result in JSON."
        )

        return TaskInstance(
            task_id=task_id,
            task_type=self.task_type,
            instruction=instruction,
            input_data=input_data,
            metadata=metadata,
        )

    def evaluate(self, model_output: Dict[str, Any], task_instance: TaskInstance) -> TaskResult:
        gt = task_instance.metadata.get("ground_truth", None)

        if not isinstance(gt, list) or len(gt) == 0:
            return TaskResult(
                task_id=task_instance.task_id,
                task_type=task_instance.task_type,
                success=False,
                score=0.0,
                metrics={},
                feedback={"error": "Ground truth box status is invalid.", "ground_truth": gt},
                raw_output=model_output,
            )

        pred_boxes = model_output.get("boxes", None)
        if not isinstance(pred_boxes, list):
            return TaskResult(
                task_id=task_instance.task_id,
                task_type=task_instance.task_type,
                success=False,
                score=0.0,
                metrics={},
                feedback={
                    "error": "Model output must contain a 'boxes' field with a list of box predictions.",
                    "ground_truth": gt,
                },
                raw_output=model_output,
            )

        def normalize_box_key(box_value):
            if isinstance(box_value, (list, tuple)) and len(box_value) == 2:
                return tuple(box_value)
            return None

        def normalize_push_list(pushes):
            if not isinstance(pushes, list):
                return []
            allowed = {"up", "down", "left", "right"}
            normalized = []
            for p in pushes:
                if isinstance(p, str):
                    v = p.strip().lower()
                    if v in allowed:
                        normalized.append(v)
            return sorted(set(normalized))

        gt_by_box = {}
        for entry in gt:
            if not isinstance(entry, dict):
                continue
            box_key = normalize_box_key(entry.get("box"))
            if box_key is None:
                continue

            gt_legal_pushes = normalize_push_list(entry.get("legal_pushes", []))
            gt_by_box[box_key] = {
                "on_target": bool(entry.get("on_target", False)),
                "on_static_dead_square": bool(entry.get("on_static_dead_square", False)),
                "in_blocked_2x2": bool(entry.get("in_blocked_2x2", False)),
                "currently_immovable": len(gt_legal_pushes) == 0,
                "legal_pushes": gt_legal_pushes,
            }

        pred_by_box = {}
        invalid_pred_entries = []
        for entry in pred_boxes:
            if not isinstance(entry, dict):
                invalid_pred_entries.append(entry)
                continue

            box_key = normalize_box_key(entry.get("box"))
            if box_key is None:
                invalid_pred_entries.append(entry)
                continue

            pred_by_box[box_key] = {
                "on_target": _safe_bool(entry.get("on_target")),
                "on_static_dead_square": _safe_bool(entry.get("on_static_dead_square")),
                "in_blocked_2x2": _safe_bool(entry.get("in_blocked_2x2")),
                "currently_immovable": _safe_bool(entry.get("currently_immovable")),
                "legal_pushes": normalize_push_list(entry.get("legal_pushes", [])),
            }

        gt_boxes = set(gt_by_box.keys())

        on_target_correct = 0
        on_static_dead_square_correct = 0
        in_blocked_2x2_correct = 0
        currently_immovable_correct = 0
        legal_pushes_correct = 0

        per_box_results = {}

        for box in sorted(gt_boxes):
            gt_entry = gt_by_box[box]
            pred_entry = pred_by_box.get(box, None)

            if pred_entry is None:
                per_box_results[str(box)] = {
                    "matched": False,
                    "expected": gt_entry,
                    "predicted": None,
                }
                continue

            on_target_ok = (
                pred_entry["on_target"] is not None and
                pred_entry["on_target"] == gt_entry["on_target"]
            )
            on_static_dead_square_ok = (
                pred_entry["on_static_dead_square"] is not None and
                pred_entry["on_static_dead_square"] == gt_entry["on_static_dead_square"]
            )
            in_blocked_2x2_ok = (
                pred_entry["in_blocked_2x2"] is not None and
                pred_entry["in_blocked_2x2"] == gt_entry["in_blocked_2x2"]
            )
            currently_immovable_ok = (
                pred_entry["currently_immovable"] is not None and
                pred_entry["currently_immovable"] == gt_entry["currently_immovable"]
            )
            legal_pushes_ok = pred_entry["legal_pushes"] == gt_entry["legal_pushes"]

            if on_target_ok:
                on_target_correct += 1
            if on_static_dead_square_ok:
                on_static_dead_square_correct += 1
            if in_blocked_2x2_ok:
                in_blocked_2x2_correct += 1
            if currently_immovable_ok:
                currently_immovable_correct += 1
            if legal_pushes_ok:
                legal_pushes_correct += 1

            per_box_results[str(box)] = {
                "matched": True,
                "on_target_correct": on_target_ok,
                "on_static_dead_square_correct": on_static_dead_square_ok,
                "in_blocked_2x2_correct": in_blocked_2x2_ok,
                "currently_immovable_correct": currently_immovable_ok,
                "legal_pushes_correct": legal_pushes_ok,
                "expected": gt_entry,
                "predicted": pred_entry,
            }

        num_gt_boxes = len(gt_boxes)

        on_target_accuracy = on_target_correct / num_gt_boxes if num_gt_boxes > 0 else 0.0
        on_static_dead_square_accuracy = on_static_dead_square_correct / num_gt_boxes if num_gt_boxes > 0 else 0.0
        in_blocked_2x2_accuracy = in_blocked_2x2_correct / num_gt_boxes if num_gt_boxes > 0 else 0.0
        currently_immovable_accuracy = currently_immovable_correct / num_gt_boxes if num_gt_boxes > 0 else 0.0
        legal_pushes_accuracy = legal_pushes_correct / num_gt_boxes if num_gt_boxes > 0 else 0.0

        score = (
            0.2 * on_target_accuracy +
            0.2 * on_static_dead_square_accuracy +
            0.2 * in_blocked_2x2_accuracy +
            0.2 * currently_immovable_accuracy +
            0.2 * legal_pushes_accuracy
        )

        success = score == 1.0

        feedback = {
            "invalid_prediction_entries": invalid_pred_entries,
            "per_box_results": per_box_results,
            "ground_truth": gt,
        }

        metrics = {
            "on_target_accuracy": on_target_accuracy,
            "on_static_dead_square_accuracy": on_static_dead_square_accuracy,
            "in_blocked_2x2_accuracy": in_blocked_2x2_accuracy,
            "currently_immovable_accuracy": currently_immovable_accuracy,
            "legal_pushes_accuracy": legal_pushes_accuracy,
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