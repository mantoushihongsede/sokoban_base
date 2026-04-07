import collections
from typing import Any, Deque, Dict, List, Set

from core.state import SokobanMap, SokobanState, Pos
from .base_actions import directions, is_floor, is_free_floor, list_legal_pushes


def currently_immovable_boxes(sokoban_map: SokobanMap, state: SokobanState) -> Dict[str, Any]:
    pushes = list_legal_pushes(sokoban_map, state)["legal_pushes"]
    pushable_boxes = {tuple(a["box_from"]) for a in pushes}

    immovable = sorted([box for box in state.boxes if box not in pushable_boxes])

    return {
        "immovable_boxes": [list(p) for p in immovable],
        "count": len(immovable),
        "note": "These boxes currently have no legal push, but this does not guarantee unsolvability.",
    }


def heuristic_no_legal_push(sokoban_map: SokobanMap, state: SokobanState) -> Dict[str, Any]:
    solved = state.is_solved(sokoban_map)
    push_count = list_legal_pushes(sokoban_map, state)["count"]

    if solved:
        return {
            "triggered": False,
            "reasons": [],
            "positions": [],
        }

    if push_count == 0:
        return {
            "triggered": True,
            "reasons": ["The current state has no legal push actions."],
            "positions": [],
        }

    return {
        "triggered": False,
        "reasons": [],
        "positions": [],
    }


def heuristic_local_mobility(sokoban_map: SokobanMap, state: SokobanState) -> Dict[str, Any]:
    box_set = set(state.boxes)
    warnings: List[Dict[str, Any]] = []
    warned_positions: Set[Pos] = set()

    for box in sorted(box_set):
        if box in sokoban_map.targets:
            continue

        br, bc = box
        free_neighbors = 0
        for dr, dc in directions():
            nxt = (br + dr, bc + dc)
            if is_free_floor(sokoban_map, nxt, box_set):
                free_neighbors += 1

        if free_neighbors == 0:
            warnings.append({
                "box": list(box),
                "severity": "high",
                "reason": "Box has no adjacent free floor tiles.",
                "free_neighbor_count": 0,
            })
            warned_positions.add(box)
        elif free_neighbors == 1:
            warnings.append({
                "box": list(box),
                "severity": "medium",
                "reason": "Box has very limited local mobility.",
                "free_neighbor_count": 1,
            })
            warned_positions.add(box)

    return {
        "triggered": len(warnings) > 0,
        "warnings": warnings,
        "positions": [list(p) for p in sorted(warned_positions)],
    }


def heuristic_region_pressure(sokoban_map: SokobanMap, state: SokobanState) -> Dict[str, Any]:
    visited_floor: Set[Pos] = set()
    box_set = set(state.boxes)
    region_warnings: List[Dict[str, Any]] = []
    warned_positions: Set[Pos] = set()

    for r in range(sokoban_map.height):
        for c in range(sokoban_map.width):
            start = (r, c)

            if start in visited_floor:
                continue
            if not is_floor(sokoban_map, start):
                continue
            if start in box_set:
                continue

            q: Deque[Pos] = collections.deque([start])
            visited_floor.add(start)
            region_tiles: Set[Pos] = {start}
            boundary_boxes: Set[Pos] = set()
            targets_in_region = 0

            while q:
                curr = q.popleft()
                if curr in sokoban_map.targets:
                    targets_in_region += 1

                cr, cc = curr
                for dr, dc in directions():
                    nxt = (cr + dr, cc + dc)

                    if not sokoban_map.in_bounds(nxt):
                        continue
                    if nxt in sokoban_map.walls:
                        continue

                    if nxt in box_set:
                        if nxt not in sokoban_map.targets:
                            boundary_boxes.add(nxt)
                        continue

                    if nxt not in visited_floor:
                        visited_floor.add(nxt)
                        region_tiles.add(nxt)
                        q.append(nxt)

            if boundary_boxes and len(boundary_boxes) > targets_in_region:
                warned_positions.update(boundary_boxes)
                region_warnings.append({
                    "region_tile_count": len(region_tiles),
                    "targets_in_region": targets_in_region,
                    "boundary_non_goal_boxes": len(boundary_boxes),
                    "boundary_boxes": [list(p) for p in sorted(boundary_boxes)],
                    "reason": (
                        "A free-floor region is adjacent to more non-goal boxes "
                        "than targets inside the region."
                    ),
                })

    return {
        "triggered": len(region_warnings) > 0,
        "warnings": region_warnings,
        "positions": [list(p) for p in sorted(warned_positions)],
    }


def get_heuristic_warnings(sokoban_map: SokobanMap, state: SokobanState) -> Dict[str, Any]:
    reports = {
        "no_legal_push": heuristic_no_legal_push(sokoban_map, state),
        "local_mobility": heuristic_local_mobility(sokoban_map, state),
        "region_pressure": heuristic_region_pressure(sokoban_map, state),
    }

    all_positions: Set[Pos] = set()
    all_reasons: List[str] = []

    for rep in reports.values():
        for p in rep.get("positions", []):
            all_positions.add((p[0], p[1]))

        for reason in rep.get("reasons", []):
            all_reasons.append(reason)

        for warning in rep.get("warnings", []):
            if "reason" in warning:
                all_reasons.append(warning["reason"])

    all_reasons = sorted(set(all_reasons))

    return {
        "has_warning": any(rep.get("triggered", False) for rep in reports.values()),
        "warning_reasons": all_reasons,
        "warning_positions": [list(p) for p in sorted(all_positions)],
        "details": reports,
    }