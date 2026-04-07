import json
import os
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

from analyzers.tools import SokobanAnalysisTools
from analyzers.tool_schemas import get_openai_tools_by_names, ALL_BASIC_TOOL_NAMES
from analyzers.json_utils import json_to_state, full_state_to_json
from llm_protocol.function_call_runner import FunctionCallRunner
from llm_protocol.openai_client import OpenAIClient

from core.state import SokobanMap, SokobanState
from core.serializer import render_text_map


def build_sokoban_objects(raw_item: dict) -> Tuple[SokobanMap, SokobanState]:
    sokoban_map = SokobanMap(
        level_id=raw_item["level_id"],
        height=raw_item["height"],
        width=raw_item["width"],
        walls=frozenset(tuple(x) for x in raw_item["walls"]),
        targets=frozenset(tuple(x) for x in raw_item["targets"]),
    )
    state = SokobanState(
        agent_pos=tuple(raw_item["agent_pos"]),
        boxes=frozenset(tuple(x) for x in raw_item["boxes"]),
    )
    return sokoban_map, state


def canonical_state_key(state: SokobanState) -> Tuple[Any, ...]:
    return (
        tuple(state.agent_pos),
        tuple(sorted(state.boxes)),
    )


def normalize_action_json(action: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "box_from": [int(action["box_from"][0]), int(action["box_from"][1])],
        "box_to": [int(action["box_to"][0]), int(action["box_to"][1])],
        "player_from": [int(action["player_from"][0]), int(action["player_from"][1])],
        "direction": [int(action["direction"][0]), int(action["direction"][1])],
    }


def canonical_action_key(action: Dict[str, Any]) -> Tuple[Any, ...]:
    a = normalize_action_json(action)
    return (
        tuple(a["box_from"]),
        tuple(a["box_to"]),
        tuple(a["player_from"]),
        tuple(a["direction"]),
    )


def safe_parse_final_action(final_text: str) -> Optional[Dict[str, Any]]:
    """
    期望模型输出:
    {
      "chosen_action": {...},
      "reason": "..."
    }
    """
    try:
        data = json.loads(final_text)
        if not isinstance(data, dict):
            return None
        action = data.get("chosen_action")
        if not isinstance(action, dict):
            return None
        return normalize_action_json(action)
    except Exception:
        return None


def make_system_prompt(per_turn_tool_budget: int) -> str:
    return f"""
You are a Sokoban push-planning agent.

You are solving the full level across multiple turns.
In THIS turn, your task is to choose exactly ONE best next legal push action for the CURRENT state.

You may call analysis tools, but you must stay concise and efficient.
Per-turn tool budget: at most {per_turn_tool_budget} tool calls.

Important rules:
1. Focus only on the CURRENT state.
2. Use tools to inspect legal pushes and detect deadlocks.
3. Choose exactly one legal push action.
4. Avoid repeating recent actions or returning to repeated states when possible.
5. Return ONLY a JSON object.

Required output format:
{{
  "chosen_action": {{
    "box_from": [row, col],
    "box_to": [row, col],
    "player_from": [row, col],
    "direction": [dr, dc]
  }},
  "reason": "short explanation"
}}
""".strip()


def make_user_prompt(
    sokoban_map: SokobanMap,
    state: SokobanState,
    recent_actions: List[Dict[str, Any]],
    turn_index: int,
    max_turns: int,
    repeated_state_count: int,
    repeated_action_count: int,
) -> str:
    payload = {
        "turn_index": turn_index,
        "remaining_turn_budget": max_turns - turn_index,
        "map": {
            "id": sokoban_map.level_id,
            "height": sokoban_map.height,
            "width": sokoban_map.width,
            "wall_positions": sorted([list(w) for w in sokoban_map.walls]),
            "target_positions": sorted([list(t) for t in sokoban_map.targets]),
        },
        "state": {
            "player_pos": list(state.agent_pos),
            "box_positions": sorted([list(b) for b in state.boxes]),
        },
        "recent_actions": recent_actions[-3:],
        "loop_warning": {
            "repeated_state_count": repeated_state_count,
            "repeated_action_count": repeated_action_count,
            "message": (
                "Recent repetition detected. Avoid undoing progress or repeating the same push pattern."
                if repeated_state_count > 0 or repeated_action_count > 0
                else "No recent repetition warning."
            ),
        },
        "text_map": render_text_map(sokoban_map, state),
    }

    return (
        "Current Sokoban turn input:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Use tools if needed, choose the best next push, and return ONLY the required JSON object."
    )


def reverse_push_pair(a1: Dict[str, Any], a2: Dict[str, Any]) -> bool:
    """
    检测两个 push 是否互为直接反向：
    a1: box_from -> box_to
    a2: box_from == a1.box_to 且 box_to == a1.box_from
    """
    try:
        x1 = normalize_action_json(a1)
        x2 = normalize_action_json(a2)
        return (
            x1["box_from"] == x2["box_to"]
            and x1["box_to"] == x2["box_from"]
        )
    except Exception:
        return False


def box_layout_key(state: SokobanState) -> Tuple[Any, ...]:
    """
    只看箱子布局，不看玩家位置。
    用于判断是否长期没有真正推进箱子局面。
    """
    return tuple(sorted(state.boxes))


def run_episode(
    runner: FunctionCallRunner,
    tools: SokobanAnalysisTools,
    sokoban_map: SokobanMap,
    init_state: SokobanState,
    max_turns: int = 30,
    max_repeat_state: int = 3,
    max_repeat_box_layout: int = 4,
    max_consecutive_reverse_pushes: int = 2,
    max_stagnation_turns: int = 6,
) -> Dict[str, Any]:
    """
    增强版：
    1. 外部多轮驱动，LLM 每轮只选一个 push
    2. 只给模型当前状态 + 最近3步 + 极简警告
    3. 检测重复状态
    4. 检测重复箱子布局
    5. 检测反向 push 振荡
    6. 检测连续无进展并提前终止
    """

    state = init_state
    action_history: List[Dict[str, Any]] = []
    turn_logs: List[Dict[str, Any]] = []

    seen_states: Dict[Tuple[Any, ...], int] = {}
    seen_box_layouts: Dict[Tuple[Any, ...], int] = {}
    seen_actions: Dict[Tuple[Any, ...], int] = {}

    consecutive_reverse_pushes = 0
    stagnation_turns = 0

    def build_loop_warning(
        current_state: SokobanState,
        action_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        state_key = canonical_state_key(current_state)
        layout_key = box_layout_key(current_state)

        repeated_state_count = seen_states.get(state_key, 0)
        repeated_box_layout_count = seen_box_layouts.get(layout_key, 0)

        repeated_action_count = 0
        if action_history:
            last_action_key = canonical_action_key(action_history[-1])
            repeated_action_count = seen_actions.get(last_action_key, 0)

        warning_messages = []

        if repeated_state_count > 0:
            warning_messages.append(
                f"Current full state has appeared before {repeated_state_count} time(s)."
            )

        if repeated_box_layout_count > 0:
            warning_messages.append(
                f"Current box layout has appeared before {repeated_box_layout_count} time(s)."
            )

        if repeated_action_count > 0:
            warning_messages.append(
                f"The most recent action pattern has appeared before {repeated_action_count} time(s)."
            )

        if consecutive_reverse_pushes > 0:
            warning_messages.append(
                f"Recent reverse-push oscillation detected {consecutive_reverse_pushes} time(s)."
            )

        if stagnation_turns > 0:
            warning_messages.append(
                f"No clear box-layout progress for {stagnation_turns} turn(s)."
            )

        if not warning_messages:
            warning_messages.append("No recent repetition warning.")

        return {
            "repeated_state_count": repeated_state_count,
            "repeated_box_layout_count": repeated_box_layout_count,
            "repeated_action_count": repeated_action_count,
            "consecutive_reverse_pushes": consecutive_reverse_pushes,
            "stagnation_turns": stagnation_turns,
            "message": " ".join(warning_messages),
        }

    def make_enhanced_user_prompt(
        sokoban_map: SokobanMap,
        state: SokobanState,
        action_history: List[Dict[str, Any]],
        turn_index: int,
        max_turns: int,
    ) -> str:
        payload = {
            "turn_index": turn_index,
            "remaining_turn_budget": max_turns - turn_index,
            "map": {
                "id": sokoban_map.level_id,
                "height": sokoban_map.height,
                "width": sokoban_map.width,
                "wall_positions": sorted([list(w) for w in sokoban_map.walls]),
                "target_positions": sorted([list(t) for t in sokoban_map.targets]),
            },
            "state": {
                "player_pos": list(state.agent_pos),
                "box_positions": sorted([list(b) for b in state.boxes]),
            },
            "recent_actions": action_history[-3:],
            "loop_warning": build_loop_warning(state, action_history),
            "text_map": render_text_map(sokoban_map, state),
        }

        return (
            "Current Sokoban turn input:\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
            "Use tools if needed, choose the best next push, and return ONLY the required JSON object."
        )

    for turn_idx in range(max_turns):
        if state.is_solved(sokoban_map):
            return {
                "status": "solved",
                "turns_used": turn_idx,
                "final_state": full_state_to_json(sokoban_map, state),
                "action_history": action_history,
                "turn_logs": turn_logs,
                "stop_reason": "Puzzle solved.",
            }

        # 先看当前状态是否已反复过多，提前告警但不立即停
        current_state_key = canonical_state_key(state)
        current_box_layout_key = box_layout_key(state)

        current_state_repeat = seen_states.get(current_state_key, 0)
        current_layout_repeat = seen_box_layouts.get(current_box_layout_key, 0)

        if current_state_repeat >= max_repeat_state:
            turn_logs.append({
                "turn_index": turn_idx,
                "input_state": full_state_to_json(sokoban_map, state),
                "warning": f"Current full state already repeated {current_state_repeat} times before this turn.",
            })

        if current_layout_repeat >= max_repeat_box_layout:
            turn_logs.append({
                "turn_index": turn_idx,
                "input_state": full_state_to_json(sokoban_map, state),
                "warning": f"Current box layout already repeated {current_layout_repeat} times before this turn.",
            })

        system_prompt = make_system_prompt(
            per_turn_tool_budget=getattr(runner, "max_tool_calls", 999)
        )

        user_prompt = make_enhanced_user_prompt(
            sokoban_map=sokoban_map,
            state=state,
            action_history=action_history,
            turn_index=turn_idx,
            max_turns=max_turns,
        )

        result = runner.run(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        final_text = result["final_text"]
        chosen_action = safe_parse_final_action(final_text)

        turn_log = {
            "turn_index": turn_idx,
            "input_state": full_state_to_json(sokoban_map, state),
            "llm_final_text": final_text,
            "runner_logs": result.get("logs", []),
            "tool_call_count": result.get("tool_call_count", 0),
            "runner_error": result.get("error"),
            "loop_warning": build_loop_warning(state, action_history),
        }

        # runner 层预算耗尽 or 其他错误
        if result.get("error") and chosen_action is None:
            turn_log["error"] = f"Runner stopped before valid action was produced: {result['error']}"
            turn_logs.append(turn_log)
            return {
                "status": "failed",
                "reason": f"Runner stopped early: {result['error']}",
                "turns_used": turn_idx,
                "final_state": full_state_to_json(sokoban_map, state),
                "action_history": action_history,
                "turn_logs": turn_logs,
                "stop_reason": "Runner error before valid action.",
            }

        if chosen_action is None:
            turn_log["error"] = "Failed to parse chosen_action from model output."
            turn_logs.append(turn_log)
            return {
                "status": "failed",
                "reason": "Model output is not valid action JSON.",
                "turns_used": turn_idx,
                "final_state": full_state_to_json(sokoban_map, state),
                "action_history": action_history,
                "turn_logs": turn_logs,
                "stop_reason": "Invalid model output.",
            }

        apply_result = tools.apply_push(sokoban_map, state, chosen_action)
        turn_log["chosen_action"] = chosen_action
        turn_log["apply_result"] = apply_result

        if not apply_result.get("ok", False):
            turn_log["error"] = "Chosen action is illegal or failed during apply_push."
            turn_logs.append(turn_log)
            return {
                "status": "failed",
                "reason": "Model chose an invalid push.",
                "turns_used": turn_idx,
                "final_state": full_state_to_json(sokoban_map, state),
                "action_history": action_history,
                "turn_logs": turn_logs,
                "stop_reason": "Illegal chosen action.",
            }

        next_state = json_to_state(apply_result["new_state"])
        next_state_key = canonical_state_key(next_state)
        next_box_layout_key = box_layout_key(next_state)

        # 反向 push 检测
        if action_history and reverse_push_pair(action_history[-1], chosen_action):
            consecutive_reverse_pushes += 1
        else:
            consecutive_reverse_pushes = 0

        # 无进展检测：如果箱子布局重复，就算一种 stagnation
        if seen_box_layouts.get(next_box_layout_key, 0) > 0:
            stagnation_turns += 1
        else:
            stagnation_turns = 0

        # 记录“当前动作”和“到达的新状态”
        action_history.append(chosen_action)

        action_key = canonical_action_key(chosen_action)
        seen_actions[action_key] = seen_actions.get(action_key, 0) + 1

        seen_states[next_state_key] = seen_states.get(next_state_key, 0) + 1
        seen_box_layouts[next_box_layout_key] = seen_box_layouts.get(next_box_layout_key, 0) + 1

        turn_log["next_state"] = full_state_to_json(sokoban_map, next_state)
        turn_log["reverse_push_detected"] = (
            len(action_history) >= 2 and reverse_push_pair(action_history[-2], action_history[-1])
        )
        turn_log["consecutive_reverse_pushes"] = consecutive_reverse_pushes
        turn_log["stagnation_turns"] = stagnation_turns
        turn_log["next_state_repeat_count"] = seen_states.get(next_state_key, 0)
        turn_log["next_box_layout_repeat_count"] = seen_box_layouts.get(next_box_layout_key, 0)

        turn_logs.append(turn_log)
        state = next_state

        # 成功后立刻检查 solved
        if state.is_solved(sokoban_map):
            return {
                "status": "solved",
                "turns_used": turn_idx + 1,
                "final_state": full_state_to_json(sokoban_map, state),
                "action_history": action_history,
                "turn_logs": turn_logs,
                "stop_reason": "Puzzle solved after applying chosen action.",
            }

        # 提前终止条件 1：反向推过多
        if consecutive_reverse_pushes >= max_consecutive_reverse_pushes:
            return {
                "status": "partial",
                "reason": "Stopped due to repeated reverse-push oscillation.",
                "turns_used": turn_idx + 1,
                "final_state": full_state_to_json(sokoban_map, state),
                "action_history": action_history,
                "turn_logs": turn_logs,
                "stop_reason": (
                    f"consecutive_reverse_pushes={consecutive_reverse_pushes} "
                    f">= max_consecutive_reverse_pushes={max_consecutive_reverse_pushes}"
                ),
            }

        # 提前终止条件 2：箱子布局长期无进展
        if stagnation_turns >= max_stagnation_turns:
            return {
                "status": "partial",
                "reason": "Stopped due to prolonged stagnation in box layout.",
                "turns_used": turn_idx + 1,
                "final_state": full_state_to_json(sokoban_map, state),
                "action_history": action_history,
                "turn_logs": turn_logs,
                "stop_reason": (
                    f"stagnation_turns={stagnation_turns} "
                    f">= max_stagnation_turns={max_stagnation_turns}"
                ),
            }

    return {
        "status": "partial" if not state.is_solved(sokoban_map) else "solved",
        "reason": "Turn budget exhausted.",
        "turns_used": max_turns,
        "final_state": full_state_to_json(sokoban_map, state),
        "action_history": action_history,
        "turn_logs": turn_logs,
        "stop_reason": f"Reached max_turns={max_turns}.",
    }


def main():
    raw_sample = {
        "level_id": "level_1",
        "txt_filename": "M100.txt",
        "image_path": "order_medium\\level_1.png",
        "height": 6,
        "width": 6,
        "walls": [
            [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
            [1, 0], [1, 2], [1, 5],
            [2, 0], [2, 5],
            [3, 0], [3, 1], [3, 2], [3, 5],
            [4, 0], [4, 1], [4, 5],
            [5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5]
        ],
        "targets": [[4, 2], [4, 4]],
        "agent_pos": [1, 1],
        "boxes": [[2, 2], [4, 3]],
    }

    sokoban_map, init_state = build_sokoban_objects(raw_sample)

    provider = "openai"
    load_dotenv(f"env\\.env.{provider}", override=True)
    api_key = os.getenv("API_KEY")
    url = os.getenv("BASE_URL")

    client = OpenAIClient(
        api_key=api_key,
        base_url=url,
    )

    tools = SokobanAnalysisTools()

    runner = FunctionCallRunner(
        client=client,
        model_name="gpt-4.1-mini",
        tools=tools,
        openai_tools=get_openai_tools_by_names(ALL_BASIC_TOOL_NAMES),
        max_rounds=8,   # 每轮 function-calling 轮数
        max_tool_calls=6,  # 每轮 function-calling 中允许的最大工具调用次数
    )

    result = run_episode(
        runner=runner,
        tools=tools,
        sokoban_map=sokoban_map,
        init_state=init_state,
        max_turns=30,
        max_repeat_state=3,
        max_repeat_box_layout=4,
        max_consecutive_reverse_pushes=2,
        max_stagnation_turns=6,
    )

    print("===== EPISODE RESULT =====")
    print(json.dumps({
        "status": result["status"],
        "reason": result.get("reason"),
        "turns_used": result["turns_used"],
        "action_history": result["action_history"],
        "final_state": result["final_state"],
    }, ensure_ascii=False, indent=2))

    print("\n===== TURN LOGS =====")
    print(json.dumps(result["turn_logs"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()