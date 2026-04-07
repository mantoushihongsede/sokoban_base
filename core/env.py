from dataclasses import dataclass
from typing import Optional, Dict, Any
from .state import SokobanMap, SokobanState, Pos
from .actions import Action, ACTION_MAP

#修改一定要慎重，后面有很多调用
@dataclass
class StepResult:
    success: bool
    message: str
    prev_state: SokobanState
    next_state: SokobanState
    action: str #这里想一下该是什么类型
    pushed_box: bool = False
    pushed_from: Optional[Pos] = None
    pushed_to: Optional[Pos] = None

    #这部分后期考虑是否应该放在这里
    # is_solved: bool
    # is_deadlock: bool
    # deadlock_reason: Optional[str] = None


class SokobanEnv:
    def __init__(self, sokoban_map: SokobanMap, init_state: SokobanState):
        self.map = sokoban_map
        self.state = init_state

    def clone(self):
        return SokobanEnv(self.map, self.state)

    def reset(self, state: SokobanState):
        self.state = state

    def is_legal_move(self, action: Action) -> bool:
        result = self.simulate_step(action)
        return result.success

    def simulate_step(self, action: Action) -> StepResult:
        if action not in ACTION_MAP:
            return StepResult(
                success=False,
                message=f"Invalid action: {action}",
                prev_state=self.state,
                next_state=self.state,
                action=action,
            )

        dr, dc = ACTION_MAP[action]
        ar, ac = self.state.agent_pos
        nr, nc = ar + dr, ac + dc
        next_pos = (nr, nc)

        if not self.map.in_bounds(next_pos):
            return StepResult(success=False, message="Agent move out of bounds", prev_state=self.state, next_state=self.state, action=action)

        if next_pos in self.map.walls:
            return StepResult(success=False, message="Agent hits a wall", prev_state=self.state, next_state=self.state, action=action)

        boxes = set(self.state.boxes)

        # 推箱子
        if next_pos in boxes:
            br, bc = nr + dr, nc + dc
            box_next = (br, bc)

            if not self.map.in_bounds(box_next):
                return StepResult(success=False, message="Box push out of bounds", prev_state=self.state, next_state=self.state, action=action)

            if box_next in self.map.walls:
                return StepResult(success=False, message="Box is blocked by wall", prev_state=self.state, next_state=self.state, action=action)

            if box_next in boxes:
                return StepResult(success=False, message="Box is blocked by another box", prev_state=self.state, next_state=self.state, action=action)

            boxes.remove(next_pos)
            boxes.add(box_next)
            next_state = SokobanState(agent_pos=next_pos, boxes=frozenset(boxes))

            return StepResult(
                success=True,
                message="Pushed a box",
                prev_state=self.state,
                next_state=next_state,
                action=action,
                pushed_box=True,
                pushed_from=next_pos,
                pushed_to=box_next,
            )

        # 普通移动
        next_state = SokobanState(agent_pos=next_pos, boxes=self.state.boxes)
        return StepResult(success=True, message="Moved successfully", prev_state=self.state, next_state=next_state, action=action)

    def step(self, action: Action) -> StepResult:
        result = self.simulate_step(action)
        if result.success:
            self.state = result.next_state
        return result