from core.env import SokobanEnv

def classify_case(env: SokobanEnv, action: str) -> str:
    """分类当前状态下执行 action 的具体情况"""
    result = env.simulate_step(action)
    if result.success:
        if result.pushed_box:
            return "push_legal"
        return "move_legal"

    msg = result.message.lower()
    if "wall" in msg:
        if "box" in msg:
            return "push_blocked_by_wall"
        return "hit_wall"
    if "another box" in msg:
        return "push_blocked_by_box"
    if "out of bounds" in msg:
        return "out_of_bounds"
    return "illegal_other"


