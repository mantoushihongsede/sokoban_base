from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class TaskInstance:
    task_id: str
    task_type: str
    instruction: str
    input_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    task_id: str
    task_type: str
    success: bool
    score: float
    metrics: Dict[str, Any]
    feedback: Dict[str, Any]
    raw_output: Optional[Dict[str, Any]] = None


class BaseTask:
    task_type = "base"

    def build(self, *args, **kwargs) -> TaskInstance:
        raise NotImplementedError

    def evaluate(self, model_output: Dict[str, Any], task_instance: TaskInstance) -> TaskResult:
        raise NotImplementedError