from dataclasses import dataclass, field
from typing import List, Set, Tuple, FrozenSet
Pos = Tuple[int, int]

@dataclass(frozen=True)
class SokobanMap:
    level_id: str
    height: int
    width: int
    walls: FrozenSet[Pos]
    targets: FrozenSet[Pos]

    def in_bounds(self, pos: Pos) -> bool:
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width

@dataclass(frozen=True)
class SokobanState:
    agent_pos: Pos
    boxes: FrozenSet[Pos]

    #这函数定义在这里合适吗？
    def is_solved(self, sokoban_map: SokobanMap) -> bool:
        return all(box in sokoban_map.targets for box in self.boxes)