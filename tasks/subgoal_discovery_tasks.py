from typing import Dict, Any, List

from .base import BaseTask, TaskInstance, TaskResult
from core.state import SokobanMap, SokobanState
from core.serializer import serialize_full_state, render_text_map
from .utils.planning_case_utils import normalize_subgoal_item
from .utils.planning_ground_truth import discover_candidate_subgoals


class CandidateSubgoalDiscoveryTask(BaseTask):
    task_type = "candidate_subgoal_discovery"

    def build(
        self,
        task_id: str,
        sokoban_map: SokobanMap,
        state: SokobanState,
        source: str = "manual",
    ) -> TaskInstance:
        gold_subgoals = discover_candidate_subgoals(sokoban_map, state)
        return TaskInstance(
            task_id=task_id,
            task_type=self.task_type,
            instruction=(
                "分析当前 Sokoban 状态，给出若干合理的候选子目标，"
                "并为每个子目标标注优先级。"
            ),
            input_data={
                "map_state": serialize_full_state(sokoban_map, state),
                "text_map": render_text_map(sokoban_map, state),
            },
            metadata={
                "level_id": sokoban_map.level_id,
                "source": source,
                "gold_subgoals": gold_subgoals,
                "num_boxes": len(state.boxes),
                "num_targets": len(sokoban_map.targets),
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

        pred_items = model_output.get("subgoals", [])
        if not isinstance(pred_items, list):
            pred_items = []

        normalized = []
        for item in pred_items:
            x = normalize_subgoal_item(item)
            if x is not None:
                normalized.append(x)

        gold_subgoals = task_instance.metadata["gold_subgoals"]

        pred_signatures = set()
        for sg in normalized:
            pred_signatures.add((
                sg.get("type"),
                tuple(sg.get("object", [])) if "object" in sg else None,
                tuple(sg.get("target", [])) if "target" in sg else None,
            ))

        gold_signatures = set()
        for sg in gold_subgoals:
            gold_signatures.add((
                sg.get("type"),
                tuple(sg.get("object", [])) if "object" in sg else None,
                tuple(sg.get("target", [])) if "target" in sg else None,
            ))

        overlap = len(pred_signatures & gold_signatures)
        precision = overlap / len(pred_signatures) if pred_signatures else 0.0
        recall = overlap / len(gold_signatures) if gold_signatures else 0.0
        diversity_bonus = min(len(normalized), 3) / 3.0

        score = 0.5 * recall + 0.3 * precision + 0.2 * diversity_bonus
        success = overlap >= 1

        return TaskResult(
            task_id=task_instance.task_id,
            task_type=task_instance.task_type,
            success=success,
            score=round(score, 4),
            metrics={
                "pred_count": len(normalized),
                "gold_count": len(gold_subgoals),
                "overlap_count": overlap,
                "precision_like": round(precision, 4),
                "recall_like": round(recall, 4),
                "diversity_bonus": round(diversity_bonus, 4),
                "level_id": task_instance.metadata.get("level_id"),
                "source": task_instance.metadata.get("source"),
            },
            feedback={
                "message": "At least one reasonable subgoal found." if success else "Failed to match any reference subgoal."
            },
            raw_output=model_output,
        )