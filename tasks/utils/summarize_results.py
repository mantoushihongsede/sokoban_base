# =========================
# 通用 summary 函数
# =========================

def summarize_results(task_results):
    summary = {
        "num_samples": len(task_results),
        "success_count": 0,
        "parse_error_count": 0,
        "avg_score": 0.0,
    }

    total_score = 0.0
    for result in task_results:
        if result.success:
            summary["success_count"] += 1
        total_score += result.score

        if result.raw_output and "__parse_error__" in result.raw_output:
            summary["parse_error_count"] += 1

    if task_results:
        summary["avg_score"] = total_score / len(task_results)
        summary["success_rate"] = summary["success_count"] / len(task_results)
        summary["parse_success_rate"] = 1 - summary["parse_error_count"] / len(task_results)
    else:
        summary["success_rate"] = 0.0
        summary["parse_success_rate"] = 0.0

    return summary


def _dynamic_summarize(task_results, key_mapping=None):
    """
    内部辅助函数：动态计算 metrics 的平均值，并支持键名重映射
    """
    base = summarize_results(task_results)
    if not task_results:
        return base

    n = len(task_results)
    metric_sums = {}

    for result in task_results:
        metrics = getattr(result, "metrics", {})
        for key, value in metrics.items():
            if isinstance(value, (int, float, bool)):
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)

    key_mapping = key_mapping or {}
    
    for key, total in metric_sums.items():
        # 如果提供了映射字典并且有这个键，就用新的名字，否则默认加 avg_ 前缀
        out_key = key_mapping.get(key, f"avg_{key}")
        base[out_key] = total / n

    return base

# =========================
# 各类别的 summary 函数
# =========================

def summarize_deadlock_detection_results(task_results):
    key_mapping = {
        "is_deadlock_accuracy": "is_deadlock_accuracy", 
        "positions_exact_match": "dead_positions_exact_match_rate",
        "reason_tags_exact_match": "reason_tags_exact_match_rate",
        "reason_tags_f1": "avg_reason_tags_f1"
    }
    return _dynamic_summarize(task_results, key_mapping)


def summarize_static_dead_results(task_results):
    key_mapping = {
        "exact_match": "exact_match_rate",
        "count_match": "count_match_rate",    # 数量预测完全正确的比率
        "count_diff": "avg_count_diff",       # 数量预测的平均绝对误差
        "precision": "avg_precision",
        "recall": "avg_recall",
        "f1": "avg_f1"
    }
    return _dynamic_summarize(task_results, key_mapping)


def summarize_box_explanation_results(task_results):
    key_mapping = {
        "on_target_accuracy": "avg_on_target_accuracy",
        "on_static_dead_square_accuracy": "avg_on_static_dead_square_accuracy",
        "in_blocked_2x2_accuracy": "avg_in_blocked_2x2_accuracy",
        "currently_immovable_accuracy": "avg_currently_immovable_accuracy",
        "legal_pushes_accuracy": "avg_legal_pushes_accuracy",
    }
    return _dynamic_summarize(task_results, key_mapping)


def summarize_transition_results(task_results):
    base = summarize_results(task_results)
    n = len(task_results) if task_results else 1

    player_match_count = 0
    box_match_count = 0
    exact_match_count = 0

    for result in task_results:
        metrics = getattr(result, "metrics", {})
        p_match = metrics.get("player_match") is True
        b_match = metrics.get("box_match") is True
        
        if p_match: player_match_count += 1
        if b_match: box_match_count += 1
        if p_match and b_match: exact_match_count += 1

    base["player_accuracy"] = player_match_count / n
    base["box_accuracy"] = box_match_count / n
    base["exact_match_rate"] = exact_match_count / n
    return base


def summarize_subgoal_results(task_results):
    key_mapping = {
        "overlap_count": "avg_overlap_count",
        "recall_like": "avg_recall_like",
        "precision_like": "avg_precision_like",
        "diversity_bonus": "avg_diversity_bonus"
    }
    return _dynamic_summarize(task_results, key_mapping)


def summarize_box_priority_results(task_results):
    key_mapping = {
        "exact_match": "exact_match_rate",
        "top1_accuracy": "top1_accuracy",
        "prefix_match_ratio": "avg_prefix_match_ratio",
        "pairwise_order_score": "avg_pairwise_order_score",
    }
    return _dynamic_summarize(task_results, key_mapping)


def summarize_assignment_results(task_results):
    key_mapping = {
        "exact_match": "exact_match_rate",
        "count_match": "count_match_rate",
        "precision": "avg_precision",
        "recall": "avg_recall",
        "f1": "avg_f1",
    }
    return _dynamic_summarize(task_results, key_mapping)


def summarize_phase_results(task_results):
    key_mapping = {
        "phase_valid": "phase_label_valid_rate"
    }
    return _dynamic_summarize(task_results, key_mapping)


def summarize_subproblem_ordering_results(task_results):
    key_mapping = {
        "pairwise_order_score": "pairwise_order_avg",
        "exact_match": "exact_order_rate",
        "coverage": "coverage_avg"
    }
    return _dynamic_summarize(task_results, key_mapping)


def summarize_horizon_choice_results(task_results):
    key_mapping = {
        "choice_valid": "choice_label_valid_rate"
    }
    return _dynamic_summarize(task_results, key_mapping)