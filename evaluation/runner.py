from dataclasses import dataclass, field
from typing import List, Dict, Any

from tasks.base import TaskInstance, TaskResult, BaseTask


@dataclass
class EpisodeRecord:
    episode_id: str
    task_results: List[TaskResult] = field(default_factory=list)

    def total_score(self) -> float:
        return sum(t.score for t in self.task_results)

    def summary(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "num_tasks": len(self.task_results),
            "total_score": self.total_score(),
            "success_count": sum(1 for t in self.task_results if t.success),
        }


class EpisodeRunner:
    def __init__(self, model_callable):
        """
        model_callable: 输入 TaskInstance，输出 dict 格式答案
        """
        self.model_callable = model_callable

    def run_task(self, task: BaseTask, task_instance: TaskInstance) -> TaskResult:
        model_output = self.model_callable(task_instance)
        return task.evaluate(model_output, task_instance)

    def run_tasks(self, episode_id: str, task_and_instances: List[tuple]) -> EpisodeRecord:
        record = EpisodeRecord(episode_id=episode_id)
        for task, instance in task_and_instances:
            result = self.run_task(task, instance)
            record.task_results.append(result)
        return record