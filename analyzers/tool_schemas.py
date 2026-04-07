"""
工具层 schema。

用途：
1. 定义 analyzer/tool 的轻量输出 schema；
2. 定义 OpenAI function calling / tool calling 所需的参数 schema；
3. 供 analyzer registry / tool dispatch / 未来参数校验使用。

注意：
- 这里是“工具协议”，不是“benchmark task 输出协议”；
- 如果某个 schema 是给 LLM 任务返回格式用的，应放到 llm_protocol/schemas.py。
"""

BASIC_TOOL_NAMES_V1 = [
    "list_legal_pushes",
    "apply_push",
    "find_definite_deadlocks",
    "explain_box_status",
]

BASIC_TOOL_NAMES_V2 = [
    "list_legal_pushes",
    "apply_push",
    "find_definite_deadlocks",
    "explain_box_status",
    "get_static_dead_squares",
    "get_heuristic_warnings",
]

ALL_BASIC_TOOL_NAMES = [
    "player_reachable_tiles",
    "list_legal_pushes",
    "state_has_legal_push",
    "apply_push",
    "get_static_dead_squares",
    "boxes_on_static_dead_squares",
    "find_2x2_deadlocks",
    "find_definite_deadlocks",
    "currently_immovable_boxes",
    "heuristic_no_legal_push",
    "heuristic_local_mobility",
    "heuristic_region_pressure",
    "get_heuristic_warnings",
    "explain_box_status",
]

def get_openai_tools_by_names(names):
    allowed = set(names)
    return [
        tool for tool in OPENAI_ANALYSIS_TOOLS
        if tool["function"]["name"] in allowed
    ]


# =========================
# 轻量输出 schema
# 这些更像“工具返回值说明”
# =========================

LEGAL_PUSH_ACTION_TOOL_OUTPUT_SCHEMA = {
    "box_from": "[row, col]",
    "box_to": "[row, col]",
    "player_from": "[row, col]",
    "direction": "[dr, dc]",
}

REACHABLE_TILES_TOOL_OUTPUT_SCHEMA = {
    "player_pos": "[row, col]",
    "reachable_tiles": "[[row, col], ...]",
    "count": "int",
}

LEGAL_PUSH_LIST_TOOL_OUTPUT_SCHEMA = {
    "legal_pushes": "[LEGAL_PUSH_ACTION_TOOL_OUTPUT_SCHEMA, ...]",
    "count": "int",
}

HAS_LEGAL_PUSH_TOOL_OUTPUT_SCHEMA = {
    "has_legal_push": "bool",
    "legal_push_count": "int",
}

APPLY_PUSH_TOOL_OUTPUT_SCHEMA = {
    "ok": "bool",
    "new_state": {
        "player_pos": "[row, col]",
        "box_positions": "[[row, col], ...]",
    },
    "error": "str, optional",
    "input_action": "dict, optional",
}

STATIC_DEAD_SQUARES_TOOL_OUTPUT_SCHEMA = {
    "static_dead_squares": "[[row, col], ...]",
    "count": "int",
}

BOXES_ON_STATIC_DEAD_SQUARES_TOOL_OUTPUT_SCHEMA = {
    "boxes_on_static_dead_squares": "[[row, col], ...]",
    "count": "int",
}

DEADLOCK_2X2_TOOL_OUTPUT_SCHEMA = {
    "deadlocked_boxes": "[[row, col], ...]",
    "count": "int",
    "patterns": "[dict, ...]",
}

DEFINITE_DEADLOCK_TOOL_OUTPUT_SCHEMA = {
    "is_deadlock": "bool",
    "definite_reasons": "[str, ...]",
    "dead_positions": "[[row, col], ...]",
    "pattern_count": "int",
    "patterns": "[dict, ...]",
}

IMMOVABLE_BOXES_TOOL_OUTPUT_SCHEMA = {
    "immovable_boxes": "[[row, col], ...]",
    "count": "int",
    "note": "str",
}

HEURISTIC_WARNING_UNIT_TOOL_OUTPUT_SCHEMA = {
    "triggered": "bool",
    "reasons": "[str, ...], optional",
    "positions": "[[row, col], ...]",
    "warnings": "[dict, ...], optional",
}

HEURISTIC_WARNINGS_TOOL_OUTPUT_SCHEMA = {
    "has_warning": "bool",
    "warning_reasons": "[str, ...]",
    "warning_positions": "[[row, col], ...]",
    "details": {
        "no_legal_push": "HEURISTIC_WARNING_UNIT_TOOL_OUTPUT_SCHEMA",
        "local_mobility": "HEURISTIC_WARNING_UNIT_TOOL_OUTPUT_SCHEMA",
        "region_pressure": "HEURISTIC_WARNING_UNIT_TOOL_OUTPUT_SCHEMA",
    },
}

BOX_EXPLANATION_TOOL_OUTPUT_SCHEMA = {
    "ok": "bool",
    "box": "[row, col]",
    "on_target": "bool",
    "on_static_dead_square": "bool",
    "in_blocked_2x2": "bool",
    "local_free_neighbor_count": "int",
    "legal_pushes": "[LEGAL_PUSH_ACTION_TOOL_OUTPUT_SCHEMA, ...]",
    "warning_messages": "[str, ...]",
    "error": "str, optional",
}


# =========================
# planning tool 输出 schema
# =========================

SUBGOAL_ITEM_TOOL_OUTPUT_SCHEMA = {
    "type": "str",
    "object": "[row, col], optional",
    "target": "[row, col], optional",
    "priority": "int",
}

CANDIDATE_SUBGOAL_DISCOVERY_TOOL_OUTPUT_SCHEMA = {
    "subgoals": "[SUBGOAL_ITEM_TOOL_OUTPUT_SCHEMA, ...]",
}

BOX_PRIORITY_RANKING_TOOL_OUTPUT_SCHEMA = {
    "box_priority_order": "[[row, col], ...]",
    "reason": "str, optional",
}

BOX_TARGET_ASSIGNMENT_UNIT_TOOL_OUTPUT_SCHEMA = {
    "box": "[row, col]",
    "target": "[row, col]",
    "reason": "str, optional",
}

BOX_TARGET_ASSIGNMENT_TOOL_OUTPUT_SCHEMA = {
    "assignments": "[BOX_TARGET_ASSIGNMENT_UNIT_TOOL_OUTPUT_SCHEMA, ...]",
}

PHASE_RECOGNITION_TOOL_OUTPUT_SCHEMA = {
    "phase": "str",
    "reason": "str, optional",
}

SUBPROBLEM_ORDERING_TOOL_OUTPUT_SCHEMA = {
    "ordered_subgoals": "[str, ...]",
    "reason": "str, optional",
}

LONG_SHORT_HORIZON_CHOICE_TOOL_OUTPUT_SCHEMA = {
    "better_choice": "str",
    "reason": "str, optional",
}


# =========================
# 工具输出 schema 注册表
# 可选，用于文档或未来校验
# =========================

TOOL_OUTPUT_SCHEMA_REGISTRY = {
    "player_reachable_tiles": REACHABLE_TILES_TOOL_OUTPUT_SCHEMA,
    "list_legal_pushes": LEGAL_PUSH_LIST_TOOL_OUTPUT_SCHEMA,
    "state_has_legal_push": HAS_LEGAL_PUSH_TOOL_OUTPUT_SCHEMA,
    "apply_push": APPLY_PUSH_TOOL_OUTPUT_SCHEMA,
    "get_static_dead_squares": STATIC_DEAD_SQUARES_TOOL_OUTPUT_SCHEMA,
    "boxes_on_static_dead_squares": BOXES_ON_STATIC_DEAD_SQUARES_TOOL_OUTPUT_SCHEMA,
    "find_2x2_deadlocks": DEADLOCK_2X2_TOOL_OUTPUT_SCHEMA,
    "find_definite_deadlocks": DEFINITE_DEADLOCK_TOOL_OUTPUT_SCHEMA,
    "currently_immovable_boxes": IMMOVABLE_BOXES_TOOL_OUTPUT_SCHEMA,
    "heuristic_no_legal_push": HEURISTIC_WARNING_UNIT_TOOL_OUTPUT_SCHEMA,
    "heuristic_local_mobility": HEURISTIC_WARNING_UNIT_TOOL_OUTPUT_SCHEMA,
    "heuristic_region_pressure": HEURISTIC_WARNING_UNIT_TOOL_OUTPUT_SCHEMA,
    "get_heuristic_warnings": HEURISTIC_WARNINGS_TOOL_OUTPUT_SCHEMA,
    "explain_box_status": BOX_EXPLANATION_TOOL_OUTPUT_SCHEMA,
    "discover_candidate_subgoals": CANDIDATE_SUBGOAL_DISCOVERY_TOOL_OUTPUT_SCHEMA,
    "rank_box_priorities": BOX_PRIORITY_RANKING_TOOL_OUTPUT_SCHEMA,
    "assign_boxes_to_targets": BOX_TARGET_ASSIGNMENT_TOOL_OUTPUT_SCHEMA,
    "recognize_planning_phase": PHASE_RECOGNITION_TOOL_OUTPUT_SCHEMA,
    "order_candidate_subproblems": SUBPROBLEM_ORDERING_TOOL_OUTPUT_SCHEMA,
    "choose_better_horizon_option": LONG_SHORT_HORIZON_CHOICE_TOOL_OUTPUT_SCHEMA,
}


def get_tool_output_schema(tool_name: str):
    if tool_name not in TOOL_OUTPUT_SCHEMA_REGISTRY:
        raise ValueError(f"Unknown tool name: {tool_name}")
    return TOOL_OUTPUT_SCHEMA_REGISTRY[tool_name]


# =========================
# OpenAI function calling 输入 schema
# =========================

POSITION_ARRAY_SCHEMA = {
    "type": "array",
    "items": {"type": "integer"},
    "minItems": 2,
    "maxItems": 2,
}

POSITIONS_ARRAY_SCHEMA = {
    "type": "array",
    "items": POSITION_ARRAY_SCHEMA,
}

MAP_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "Level id."},
        "height": {"type": "integer", "description": "Map height."},
        "width": {"type": "integer", "description": "Map width."},
        "wall_positions": {
            "type": "array",
            "items": POSITION_ARRAY_SCHEMA,
            "description": "Wall positions as [row, col].",
        },
        "target_positions": {
            "type": "array",
            "items": POSITION_ARRAY_SCHEMA,
            "description": "Target positions as [row, col].",
        },
    },
    "required": ["id", "height", "width", "wall_positions", "target_positions"],
    "additionalProperties": False,
}

STATE_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "player_pos": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
            "description": "Player position as [row, col].",
        },
        "box_positions": {
            "type": "array",
            "items": POSITION_ARRAY_SCHEMA,
            "description": "Box positions as [row, col].",
        },
    },
    "required": ["player_pos", "box_positions"],
    "additionalProperties": False,
}

ACTION_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "box_from": POSITION_ARRAY_SCHEMA,
        "box_to": POSITION_ARRAY_SCHEMA,
        "player_from": POSITION_ARRAY_SCHEMA,
        "direction": POSITION_ARRAY_SCHEMA,
    },
    "required": ["box_from", "box_to", "player_from", "direction"],
    "additionalProperties": False,
}

SUBGOAL_CANDIDATE_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "type": {"type": "string"},
        "object": POSITION_ARRAY_SCHEMA,
        "target": POSITION_ARRAY_SCHEMA,
        "priority": {"type": "integer"},
    },
    "required": ["id", "type"],
    "additionalProperties": True,
}

HORIZON_OPTION_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "heuristic_risk": {"type": "integer"},
        "heuristic_progress": {"type": "integer"},
    },
    "required": ["description", "heuristic_risk", "heuristic_progress"],
    "additionalProperties": True,
}


# =========================
# OpenAI tools 定义
# =========================

OPENAI_ANALYSIS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "player_reachable_tiles",
            "description": "Return all tiles currently reachable by the player without moving any box.",
            "parameters": {
                "type": "object",
                "properties": {
                    "map": MAP_INPUT_SCHEMA,
                    "state": STATE_INPUT_SCHEMA,
                },
                "required": ["map", "state"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_legal_pushes",
            "description": "List all currently legal push actions in the given Sokoban state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "map": MAP_INPUT_SCHEMA,
                    "state": STATE_INPUT_SCHEMA,
                },
                "required": ["map", "state"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "state_has_legal_push",
            "description": "Check whether the current state has at least one legal push action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "map": MAP_INPUT_SCHEMA,
                    "state": STATE_INPUT_SCHEMA,
                },
                "required": ["map", "state"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_push",
            "description": "Apply one legal push action and return the resulting state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "map": MAP_INPUT_SCHEMA,
                    "state": STATE_INPUT_SCHEMA,
                    "action": ACTION_INPUT_SCHEMA,
                },
                "required": ["map", "state", "action"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_static_dead_squares",
            "description": "Return all static dead squares computed by reverse-push analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "map": MAP_INPUT_SCHEMA,
                },
                "required": ["map"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "boxes_on_static_dead_squares",
            "description": "Return boxes currently standing on static dead squares.",
            "parameters": {
                "type": "object",
                "properties": {
                    "map": MAP_INPUT_SCHEMA,
                    "state": STATE_INPUT_SCHEMA,
                },
                "required": ["map", "state"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_2x2_deadlocks",
            "description": "Detect blocked 2x2 deadlock structures.",
            "parameters": {
                "type": "object",
                "properties": {
                    "map": MAP_INPUT_SCHEMA,
                    "state": STATE_INPUT_SCHEMA,
                },
                "required": ["map", "state"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_definite_deadlocks",
            "description": "Detect strict deadlocks using only sound rules such as static dead squares and blocked 2x2 structures.",
            "parameters": {
                "type": "object",
                "properties": {
                    "map": MAP_INPUT_SCHEMA,
                    "state": STATE_INPUT_SCHEMA,
                },
                "required": ["map", "state"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "currently_immovable_boxes",
            "description": "Return boxes that currently have no legal push direction. This is heuristic only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "map": MAP_INPUT_SCHEMA,
                    "state": STATE_INPUT_SCHEMA,
                },
                "required": ["map", "state"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "heuristic_no_legal_push",
            "description": "Return a heuristic warning if the current state has no legal push actions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "map": MAP_INPUT_SCHEMA,
                    "state": STATE_INPUT_SCHEMA,
                },
                "required": ["map", "state"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "heuristic_local_mobility",
            "description": "Analyze local mobility risk for boxes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "map": MAP_INPUT_SCHEMA,
                    "state": STATE_INPUT_SCHEMA,
                },
                "required": ["map", "state"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "heuristic_region_pressure",
            "description": "Analyze region pressure heuristically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "map": MAP_INPUT_SCHEMA,
                    "state": STATE_INPUT_SCHEMA,
                },
                "required": ["map", "state"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_heuristic_warnings",
            "description": "Aggregate heuristic warnings about the current state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "map": MAP_INPUT_SCHEMA,
                    "state": STATE_INPUT_SCHEMA,
                },
                "required": ["map", "state"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_box_status",
            "description": "Explain the status of one box in the current state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "map": MAP_INPUT_SCHEMA,
                    "state": STATE_INPUT_SCHEMA,
                    "box": POSITION_ARRAY_SCHEMA,
                },
                "required": ["map", "state", "box"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "discover_candidate_subgoals",
            "description": "Return heuristic candidate subgoals for the current Sokoban state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "map": MAP_INPUT_SCHEMA,
                    "state": STATE_INPUT_SCHEMA,
                },
                "required": ["map", "state"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_box_priorities",
            "description": "Return a heuristic ranking of boxes to handle first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "map": MAP_INPUT_SCHEMA,
                    "state": STATE_INPUT_SCHEMA,
                },
                "required": ["map", "state"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assign_boxes_to_targets",
            "description": "Return a heuristic one-to-one assignment from boxes to targets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "map": MAP_INPUT_SCHEMA,
                    "state": STATE_INPUT_SCHEMA,
                },
                "required": ["map", "state"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recognize_planning_phase",
            "description": "Recognize the current planning phase such as clear_path or deliver_box.",
            "parameters": {
                "type": "object",
                "properties": {
                    "map": MAP_INPUT_SCHEMA,
                    "state": STATE_INPUT_SCHEMA,
                },
                "required": ["map", "state"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "order_candidate_subproblems",
            "description": "Order candidate subgoals according to heuristic dependency and risk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_subgoals": {
                        "type": "array",
                        "items": SUBGOAL_CANDIDATE_INPUT_SCHEMA,
                    },
                },
                "required": ["candidate_subgoals"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "choose_better_horizon_option",
            "description": "Choose the better option between a short-term and a long-term candidate plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "option_A": HORIZON_OPTION_INPUT_SCHEMA,
                    "option_B": HORIZON_OPTION_INPUT_SCHEMA,
                },
                "required": ["option_A", "option_B"],
                "additionalProperties": False,
            },
        },
    },
]