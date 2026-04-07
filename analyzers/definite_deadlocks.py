import collections
from typing import Any, Deque, Dict, FrozenSet, List, Optional, Set, Tuple

from core.state import SokobanMap, SokobanState, Pos
from .json_utils import positions_to_json


MapSignature = Tuple[str, int, int, FrozenSet[Pos], FrozenSet[Pos]]
Direction = Tuple[int, int]


class DefiniteDeadlockAnalyzer:
    """
    安全 oracle 层：
    - static dead square
    - reverse-push cache
    - 2x2 deadlock
    """

    def __init__(self) -> None:
        self._static_map_signature: Optional[MapSignature] = None
        self._static_dead_squares: Set[Pos] = set()

    def _directions(self) -> List[Direction]:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def _is_floor(self, sokoban_map: SokobanMap, pos: Pos) -> bool:
        return sokoban_map.in_bounds(pos) and pos not in sokoban_map.walls

    def _map_signature(self, sokoban_map: SokobanMap) -> MapSignature:
        return (
            sokoban_map.level_id,
            sokoban_map.height,
            sokoban_map.width,
            frozenset(sokoban_map.walls),
            frozenset(sokoban_map.targets),
        )

    def _ensure_static_cache(self, sokoban_map: SokobanMap) -> None:
        sig = self._map_signature(sokoban_map)
        if sig != self._static_map_signature:
            self._static_dead_squares = self._compute_static_dead_squares_internal(sokoban_map)
            self._static_map_signature = sig

    def _compute_static_dead_squares_internal(self, sokoban_map: SokobanMap) -> Set[Pos]:
        reachable_by_box: Set[Pos] = set()
        q: Deque[Pos] = collections.deque()

        for target in sokoban_map.targets:
            if self._is_floor(sokoban_map, target):
                reachable_by_box.add(target)
                q.append(target)

        while q:
            curr = q.popleft()
            cr, cc = curr

            for dr, dc in self._directions():
                prev = (cr - dr, cc - dc)
                player_pos = (cr - 2 * dr, cc - 2 * dc)

                if not self._is_floor(sokoban_map, prev):
                    continue
                if not self._is_floor(sokoban_map, player_pos):
                    continue

                if prev not in reachable_by_box:
                    reachable_by_box.add(prev)
                    q.append(prev)

        dead_squares: Set[Pos] = set()
        for r in range(sokoban_map.height):
            for c in range(sokoban_map.width):
                pos = (r, c)
                if pos in sokoban_map.walls:
                    continue
                if pos in sokoban_map.targets:
                    continue
                if pos not in reachable_by_box:
                    dead_squares.add(pos)

        return dead_squares

    def is_static_dead_square(self, sokoban_map: SokobanMap, pos: Pos) -> bool:
        self._ensure_static_cache(sokoban_map)
        return pos in self._static_dead_squares

    def get_static_dead_square_positions(self, sokoban_map: SokobanMap) -> List[Pos]:
        self._ensure_static_cache(sokoban_map)
        return sorted(self._static_dead_squares)

    def get_static_dead_squares(self, sokoban_map: SokobanMap) -> Dict[str, Any]:
        dead = self.get_static_dead_square_positions(sokoban_map)
        return {
            "static_dead_squares": positions_to_json(dead),
            "count": len(dead),
        }

    def boxes_on_static_dead_squares(self, sokoban_map: SokobanMap, state: SokobanState) -> Dict[str, Any]:
        self._ensure_static_cache(sokoban_map)
        boxes = sorted([box for box in state.boxes if box in self._static_dead_squares])
        return {
            "boxes_on_static_dead_squares": positions_to_json(boxes),
            "count": len(boxes),
        }

    def find_2x2_deadlocks(self, sokoban_map: SokobanMap, state: SokobanState) -> Dict[str, Any]:
        box_set = set(state.boxes)
        dead_boxes: Set[Pos] = set()
        patterns: List[Dict[str, Any]] = []

        for r in range(sokoban_map.height - 1):
            for c in range(sokoban_map.width - 1):
                square = [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]

                if not all((p in sokoban_map.walls or p in box_set) for p in square):
                    continue

                boxes_in_square = [p for p in square if p in box_set]
                if not boxes_in_square:
                    continue

                if all(p in sokoban_map.targets for p in boxes_in_square):
                    continue

                non_goal_boxes = [p for p in boxes_in_square if p not in sokoban_map.targets]
                if not non_goal_boxes:
                    continue

                dead_boxes.update(non_goal_boxes)

                patterns.append({
                    "type": "blocked_2x2",
                    "square": positions_to_json(square),
                    "deadlocked_boxes": positions_to_json(sorted(non_goal_boxes)),
                })

        patterns.sort(key=lambda x: (x["square"][0][0], x["square"][0][1]))
        dead_boxes_sorted = sorted(dead_boxes)

        return {
            "deadlocked_boxes": positions_to_json(dead_boxes_sorted),
            "count": len(dead_boxes_sorted),
            "patterns": patterns,
        }

    def find_definite_deadlocks(self, sokoban_map: SokobanMap, state: SokobanState) -> Dict[str, Any]:
        self._ensure_static_cache(sokoban_map)

        reasons: List[str] = []
        positions: Set[Pos] = set()
        patterns: List[Dict[str, Any]] = []

        static_boxes = sorted([box for box in state.boxes if box in self._static_dead_squares])
        for box in static_boxes:
            reasons.append(f"Box at {list(box)} is on a static dead square.")
            positions.add(box)
            patterns.append({
                "type": "static_dead_square",
                "boxes": positions_to_json([box]),
            })

        dead2x2 = self.find_2x2_deadlocks(sokoban_map, state)
        for box_json in dead2x2["deadlocked_boxes"]:
            pos = (box_json[0], box_json[1])
            reasons.append(f"Box at {list(pos)} is part of a blocked 2x2 structure.")
            positions.add(pos)

        patterns.extend(dead2x2["patterns"])

        reasons = sorted(set(reasons))
        positions_sorted = sorted(positions)

        return {
            "is_deadlock": len(reasons) > 0,
            "definite_reasons": reasons,
            "dead_positions": positions_to_json(positions_sorted),
            "pattern_count": len(patterns),
            "patterns": patterns,
        }