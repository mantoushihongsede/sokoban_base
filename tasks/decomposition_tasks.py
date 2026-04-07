from .base import BaseTask, TaskInstance, TaskResult
from analyzers.dependency import DependencyAnalyzer
from core.serializer import serialize_full_state


class SubgoalOrderingTask(BaseTask):
    task_type = "subgoal_ordering"

    def __init__(self):
        self.analyzer = DependencyAnalyzer()

    def build(self, task_id, sokoban_map, state):
        dep_result = self.analyzer.analyze(sokoban_map, state)

        return TaskInstance(
            task_id=task_id,
            task_type=self.task_type,
            instruction=(
                "请识别当前局面的合理子目标，并给出处理顺序。"
                "输出格式为一个列表，每项包含 box_pos, target_pos, priority。"
            ),
            input_data={
                "current_state": serialize_full_state(sokoban_map, state),
            },
            metadata={
                "ground_truth_subgoals": [
                    {
                        "box_pos": list(s.box_pos),
                        "target_pos": list(s.target_pos),
                        "priority": s.priority,
                        "rationale": s.rationale,
                    }
                    for s in dep_result.subgoals
                ]
            }
        )

    def evaluate(self, model_output, task_instance):
        # 这里一开始不要做得太硬
        # 先只比较“是否给出合法格式”、“是否有覆盖所有箱子”
        predicted = model_output.get("subgoals", [])
        gt = task_instance.metadata["ground_truth_subgoals"]

        success = isinstance(predicted, list)
        score = 1.0 if success else 0.0

        return TaskResult(
            task_id=task_instance.task_id,
            task_type=task_instance.task_type,
            success=success,
            score=score,
            metrics={
                "ground_truth_subgoals": gt,
                "predicted_subgoals": predicted,
            },
            feedback={
                "message": "Subgoal output parsed." if success else "Invalid subgoal output format."
            },
            raw_output=model_output,
        )