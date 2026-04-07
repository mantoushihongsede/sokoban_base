from typing import Any, Dict

from core.state import SokobanMap, SokobanState
from .json_utils import json_to_map, json_to_state, json_to_pos
from .tools import SokobanAnalysisTools


def dispatch_tool_call(
    tools: SokobanAnalysisTools,
    tool_name: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    if tool_name == "get_static_dead_squares":
        sokoban_map = json_to_map(arguments["map"])
        return tools.get_static_dead_squares(sokoban_map)
    
    # if tool_name == "order_candidate_subproblems":
    #     return tools.order_candidate_subproblems(arguments["candidate_subgoals"])

    # if tool_name == "choose_better_horizon_option":
    #     return tools.choose_better_horizon_option(arguments["option_A"], arguments["option_B"])

    sokoban_map = json_to_map(arguments["map"])
    state = json_to_state(arguments["state"])

    # if tool_name == "discover_candidate_subgoals":
    #     return tools.discover_candidate_subgoals(sokoban_map, state)

    # if tool_name == "rank_box_priorities":
    #     return tools.rank_box_priorities(sokoban_map, state)

    # if tool_name == "assign_boxes_to_targets":
    #     return tools.assign_boxes_to_targets(sokoban_map, state)

    # if tool_name == "recognize_planning_phase":
    #     return tools.recognize_planning_phase(sokoban_map, state)

    if tool_name == "player_reachable_tiles":
        return tools.player_reachable_tiles(sokoban_map, state)

    if tool_name == "list_legal_pushes":
        return tools.list_legal_pushes(sokoban_map, state)

    if tool_name == "state_has_legal_push":
        return tools.state_has_legal_push(sokoban_map, state)

    if tool_name == "apply_push":
        return tools.apply_push(sokoban_map, state, arguments["action"])

    if tool_name == "boxes_on_static_dead_squares":
        return tools.boxes_on_static_dead_squares(sokoban_map, state)

    if tool_name == "find_2x2_deadlocks":
        return tools.find_2x2_deadlocks(sokoban_map, state)

    if tool_name == "find_definite_deadlocks":
        return tools.find_definite_deadlocks(sokoban_map, state)

    if tool_name == "currently_immovable_boxes":
        return tools.currently_immovable_boxes(sokoban_map, state)

    if tool_name == "heuristic_no_legal_push":
        return tools.heuristic_no_legal_push(sokoban_map, state)

    if tool_name == "heuristic_local_mobility":
        return tools.heuristic_local_mobility(sokoban_map, state)

    if tool_name == "heuristic_region_pressure":
        return tools.heuristic_region_pressure(sokoban_map, state)

    if tool_name == "get_heuristic_warnings":
        return tools.get_heuristic_warnings(sokoban_map, state)

    if tool_name == "explain_box_status":
        box = json_to_pos(arguments["box"])
        return tools.explain_box_status(sokoban_map, state, box)

    raise ValueError(f"Unknown tool name: {tool_name}")