from tasks.box_priority_tasks import BoxPriorityRankingTask
from tasks.assignment_tasks import BoxTargetAssignmentTask
from tasks.utils.generate_deadlock_tasks import load_records_from_json


def generate_box_priority_tasks(json_path: str, num_samples: int = 20):
    task_builder = BoxPriorityRankingTask()
    records = load_records_from_json(json_path, "ground_truth_box_priority") #1

    tasks = []
    sample_id = 0

    for record in records: #2
        if len(tasks) >= num_samples:
            return tasks
        gt = record.get("ground_truth_box_priority", None) #3
        env = record.get("env", None) #4

        instance = task_builder.build(
            task_id=f"box_priority_{sample_id}",
            sokoban_map=env.map,
            state=env.state,
            case_type="box_priority",
            gt=gt, #5
        )
        tasks.append((task_builder, instance))
        sample_id += 1

    return tasks


def generate_box_target_assignment_tasks(json_path: str, num_samples: int = 20):
    task_builder = BoxTargetAssignmentTask()
    records = load_records_from_json(json_path, "ground_truth_box_target_assignment") #1

    tasks = []
    sample_id = 0

    for record in records: #2
        if len(tasks) >= num_samples:
            return tasks
        
        gt = record.get("ground_truth_box_target_assignment", None) #3
        env = record.get("env", None) #4

        instance = task_builder.build(
            task_id=f"box_target_assignment_{sample_id}",
            sokoban_map=env.map,
            state=env.state,
            case_type="box_target_assignment",
            gt=gt, #5
        )
        tasks.append((task_builder, instance))
        sample_id += 1

    return tasks