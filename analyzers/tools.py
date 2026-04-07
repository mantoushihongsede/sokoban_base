from typing import Any, Dict

from core.state import SokobanMap, SokobanState, Pos
from core.serializer import serialize_state, serialize_full_state

from .base_actions import (
    apply_push,
    list_legal_pushes,
    player_reachable_tiles,
    state_has_legal_push,
)
from .definite_deadlocks import DefiniteDeadlockAnalyzer
from .explain import explain_box_status
from .heuristic_warnings import (
    currently_immovable_boxes,
    get_heuristic_warnings,
    heuristic_local_mobility,
    heuristic_no_legal_push,
    heuristic_region_pressure,
)




class SokobanAnalysisTools:
    """
    对外统一门面。
    """

    def __init__(self) -> None:
        self.definite_deadlocks = DefiniteDeadlockAnalyzer()

    # =========================
    # JSON helpers
    # =========================

    def serialize_state(self, state: SokobanState) -> Dict[str, Any]:
        return serialize_state(state)

    def serialize_full_state(self, sokoban_map: SokobanMap, state: SokobanState) -> Dict[str, Any]:
        return serialize_full_state(sokoban_map, state)

    # =========================
    # base actions
    # =========================

    def player_reachable_tiles(self, sokoban_map: SokobanMap, state: SokobanState) -> Dict[str, Any]:
        return player_reachable_tiles(sokoban_map, state)

    def list_legal_pushes(self, sokoban_map: SokobanMap, state: SokobanState) -> Dict[str, Any]:
        return list_legal_pushes(sokoban_map, state)

    def state_has_legal_push(self, sokoban_map: SokobanMap, state: SokobanState) -> Dict[str, Any]:
        return state_has_legal_push(sokoban_map, state)

    def apply_push(self, sokoban_map: SokobanMap, state: SokobanState, action: Dict[str, Any]) -> Dict[str, Any]:
        return apply_push(sokoban_map, state, action)

    def is_goal_state(self, sokoban_map: SokobanMap, state: SokobanState) -> Dict[str, Any]:
        return {"is_goal_state": state.is_solved(sokoban_map)}

    # =========================
    # definite deadlocks
    # =========================

    def get_static_dead_squares(self, sokoban_map: SokobanMap) -> Dict[str, Any]:
        return self.definite_deadlocks.get_static_dead_squares(sokoban_map)

    def boxes_on_static_dead_squares(self, sokoban_map: SokobanMap, state: SokobanState) -> Dict[str, Any]:
        return self.definite_deadlocks.boxes_on_static_dead_squares(sokoban_map, state)

    def find_2x2_deadlocks(self, sokoban_map: SokobanMap, state: SokobanState) -> Dict[str, Any]:
        return self.definite_deadlocks.find_2x2_deadlocks(sokoban_map, state)

    def find_definite_deadlocks(self, sokoban_map: SokobanMap, state: SokobanState) -> Dict[str, Any]:
        return self.definite_deadlocks.find_definite_deadlocks(sokoban_map, state)

    # =========================
    # heuristic warnings
    # =========================

    def currently_immovable_boxes(self, sokoban_map: SokobanMap, state: SokobanState) -> Dict[str, Any]:
        return currently_immovable_boxes(sokoban_map, state)

    def heuristic_no_legal_push(self, sokoban_map: SokobanMap, state: SokobanState) -> Dict[str, Any]:
        return heuristic_no_legal_push(sokoban_map, state)

    def heuristic_local_mobility(self, sokoban_map: SokobanMap, state: SokobanState) -> Dict[str, Any]:
        return heuristic_local_mobility(sokoban_map, state)

    def heuristic_region_pressure(self, sokoban_map: SokobanMap, state: SokobanState) -> Dict[str, Any]:
        return heuristic_region_pressure(sokoban_map, state)

    def get_heuristic_warnings(self, sokoban_map: SokobanMap, state: SokobanState) -> Dict[str, Any]:
        return get_heuristic_warnings(sokoban_map, state)

    # =========================
    # explain
    # =========================

    def explain_box_status(self, sokoban_map: SokobanMap, state: SokobanState, box: Pos) -> Dict[str, Any]:
        return explain_box_status(
            sokoban_map=sokoban_map,
            state=state,
            box=box,
            deadlock_analyzer=self.definite_deadlocks,
        )
    
    # =========================
    # planning heuristics
    # =========================

    