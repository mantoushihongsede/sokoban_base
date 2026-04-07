from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class StateCompareResult:
    player_match: bool
    box_match: bool
    all_match: bool
    error_type: str

# 非常有必要、提高鲁棒性，降低对输入的要求
def normalize_positions(positions):
    return sorted([list(p) for p in positions])

# 这个输入好难受、不能是这个类型的。太蛮烦了。这里感觉再加一个message更好
def compare_predicted_state(predicted_state: Dict[str, Any], actual_state: Dict[str, Any]) -> StateCompareResult:
    predicted_player = predicted_state.get("player_pos")
    predicted_boxes = normalize_positions(predicted_state.get("box_positions", []))

    actual_player = actual_state.get("player_pos")
    actual_boxes = normalize_positions(actual_state.get("box_positions", []))

    player_match = predicted_player == actual_player
    box_match = predicted_boxes == actual_boxes
    all_match = player_match and box_match

    if all_match:
        error_type = "none"
    elif not player_match and not box_match:
        error_type = "player_and_box_mismatch"
    elif not player_match:
        error_type = "player_position_mismatch"
    else:
        error_type = "box_position_mismatch"

    return StateCompareResult(
        player_match=player_match,
        box_match=box_match,
        all_match=all_match,
        error_type=error_type,
    )