import json


def build_legality_prompt(task_instance):
    data = task_instance.input_data
    prompt = f"""
You are solving a Sokoban reasoning task.

Given the current state and a candidate action, determine whether the action is legal.

A move is LEGAL if:
   - The target cell is empty space , a target , or the agent's start position.
   - The target cell has a box AND the cell immediately behind the box in the same direction is empty space or a target.
A move is ILLEGAL if:
   - The agent moves into a wall.
   - The agent pushes a box into a wall or another box.

Return JSON only, with this format:
{{
  "legal": true
}}

Current state:
{json.dumps(data["map_state"], ensure_ascii=False)}

Candidate action:
{json.dumps(data["candidate_action"], ensure_ascii=False)}
"""
    return prompt.strip()


def build_transition_prompt(task_instance):
    data = task_instance.input_data
    prompt = f"""
You are solving a Sokoban state transition task.

Given the current state and one action, predict the next state.

Return JSON only, with this format:
{{
  "player_pos": [row, col],
  "box_positions": [[row, col], ...]
}}

Current state:
{json.dumps(data["current_state"], ensure_ascii=False)}

Action:
{json.dumps(data["action"], ensure_ascii=False)}
"""
    return prompt.strip()


def build_deadlock_detection_prompt(task_instance):
    data = task_instance.input_data
    prompt = f"""
You are solving a Sokoban definite deadlock detection task.

Determine whether the current Sokoban state is DEFINITELY deadlocked.

Definitions:
1. A state is definitely deadlocked only if it is provably unsolvable by strict structural rules.
2. "static_dead_square" means a box is on a non-target square from which no sequence of pushes can ever place that box onto any target.
3. "blocked_2x2_structure" means a box is part of a blocked  structure formed by walls and/or boxes under the task rules
4. Do NOT use heuristic or speculative reasoning such as "maybe stuck", "hard to move", "risky", or "difficult".

Output rules:
- Return valid JSON only.
- Use exactly these keys:
  - "is_deadlock": boolean
  - "definite_reasons": list
- The value of "definite_reasons" must be:
  [
    [[row, col], [reason_tag_1, reason_tag_2, ...]],
    ...
  ]
- Each box position should appear at most once.
- Allowed reason tags are only:
  - "static_dead_square"
  - "blocked_2x2_structure"

Example output for a deadlocked state:
{{
  "is_deadlock": true,
  "definite_reasons": [
    [[1, 1], ["static_dead_square", "blocked_2x2_structure"]],
    [[2, 3], ["static_dead_square"]]
  ]
}}

Example output for a non-deadlocked state:
{{
  "is_deadlock": false,
  "definite_reasons": []
}}

Current state:
{json.dumps(data["state"], ensure_ascii=False)}
"""
    return prompt.strip()
 

import json

def build_static_dead_squares_prompt(task_instance):
    data = task_instance.input_data
    prompt = f"""
You are solving a Sokoban static_dead_square identification task.

Identify all static_dead_squares on the map.

Definition:
A static_dead_square is a non-target floor tile such that, if a box is placed on that tile, there exists no legal sequence of pushes that can move that box to any target, regardless of the positions of other boxes.
This depends only on the static map structure (walls, floors, and targets), not on temporary box arrangements in the current state.

Attention:
A tile currently occupied by the agent must still be evaluated normally.
Do not exclude a floor tile from static_dead_square classification just because the agent is currently standing on it.
If a box placed on that tile could never reach any target, then it must be included.

Common examples of static dead squares:
- a non-target corner tile touching two perpendicular walls
- wall-constrained dead zones:
  non-target floor tiles where a box, once placed there, is forced by the map geometry to remain along a wall, corridor, groove, or other constrained region, and from that region can never be pushed to any target
- non-target tiles in narrow corridors, grooves, or restricted passages from which a box can never be pushed to any target

Do NOT include:
- walls
- targets
- floor tiles from which a box can still possibly reach a target
- state-dependent deadlocks caused by the current positions of boxes
- multi-box configuration deadlocks (such as 2x2 box blocks or freeze deadlocks)

Important:
The examples above are not the full definition.
Use the final reachability criterion:
if a box on that tile can never reach any target by any legal pushes, then it is a static_dead_square; otherwise it is not.

Return JSON only, with this format:
{{
  "static_dead_squares": [[row, col], ...],
  "count": <total number of static_dead_squares>
}}

Current state:
{json.dumps(data["state"], ensure_ascii=False)}
"""
    return prompt.strip()


def build_box_status_explanation_prompt(task_instance):
    data = task_instance.input_data
    prompt = f"""
You are solving a Sokoban multi-box status classification task.

Given the current Sokoban state, classify EVERY box in the state.

Coordinate System:
Coordinates are represented as [row, col]. The indices are 0-based, where:
- `row` represents the vertical axis, starting from 0 and increasing from top to bottom.
- `col` represents the horizontal axis, starting from 0 and increasing from left to right.

Definitions:
1. "on_target":
   A box is on_target if and only if its current position is a target square.

2. "on_static_dead_square":
   A box is "on_static_dead_square" if and only if it is currently on a non-target floor tile from which no sequence of legal pushes can ever move it to any target, regardless of the positions of other boxes.
   This depends only on the static map structure (walls, floors, and targets), not on temporary box arrangements in the current state.

3. "in_blocked_2x2":
   A box is in_blocked_2x2 if and only if it belongs to a blocked 2x2 structure formed by walls and/or boxes under the task rules.

4. "currently_immovable":
   A box is currently_immovable if and only if there is no legal push that can move this box in the current state.

5. "legal_pushes":
   legal_pushes is the complete list of all legal push directions for that box in the current state.
   Each direction must be one of: "up", "down", "left", "right".
   If the box cannot be legally pushed in any direction, return an empty list.


Important requirements:
- Classify ALL boxes in the current state.
- Do not omit any box.
- Do not add boxes that are not present in the current state.
- Use the box coordinates exactly as they appear in the state.
- Return valid JSON only.
- Do not return explanations, comments, or markdown.

Return JSON only, with this format:
{{
  "boxes": [
    {{
      "box": [row, col],
      "on_target": true,
      "on_static_dead_square": false,
      "in_blocked_2x2": false,
      "currently_immovable": false,
      "legal_pushes": ["left", "right"],
    }}
  ]
}}

Current state:
{json.dumps(data["state"], ensure_ascii=False)}
"""
    return prompt.strip()


def build_box_priority_prompt(task_instance):
    data = task_instance.input_data
    prompt = f"""
You are given a Sokoban state.

Task:
Return the required box-handling order for solving this level from the current state.

Definition:
- The order must list all boxes from first to last.
- The first box is the one that must be handled first.
- The ranking must reflect the required solving order.
- If the boxes are handled in a different order, the level becomes unsolvable.
- Use each box exactly once.
- Each position must be written as [row, col].

Return JSON only in this format:
{{
  "box_priority_order": [[row, col], [row, col]]
}}

Rules:
- Output JSON only.
- Do not output any text outside JSON.
- Include all boxes exactly once.
- Do not repeat boxes.
- Do not omit boxes.
- Use only the format [row, col].

Current state:
{json.dumps(data["map_state"], ensure_ascii=False)}

"""
    return prompt.strip()



def build_box_target_assignment_prompt(task_instance):
    data = task_instance.input_data
    prompt = f"""
You are given a Sokoban state.

Task:
Return the required box-to-target assignment for solving this level from the current state.

Definition:
- Assign each box to the target it must reach in a successful solution.
- The assignment must reflect the required box-target matching for solvability.
- If a box is assigned to a different target, the level becomes unsolvable.
- Output one assignment for every box.
- Each box must appear exactly once.
- Each target must appear exactly once.
- Each position must be written as [row, col].

Return JSON only in this format:
{{
  "assignments": [
    {{
      "box": [row, col],
      "target": [row, col]
    }}
  ]
}}

Rules:
- Output JSON only.
- Do not output any text outside JSON.
- Include all boxes exactly once.
- Do not repeat boxes.
- Do not repeat targets.
- Do not omit boxes.
- Use only the format [row, col].

Current state:
{json.dumps(data["map_state"], ensure_ascii=False)}

"""
    return prompt.strip()





# def build_subgoal_discovery_prompt(task_instance):
#     data = task_instance.input_data
#     prompt = f"""
# You are solving a Sokoban planning task.

# Given the current Sokoban state, propose several reasonable candidate subgoals.
# A subgoal should be a meaningful intermediate objective, not just a single primitive move.

# Allowed subgoal types:
# - clear_path
# - deliver_box
# - reposition_box
# - approach_box
# - free_target
# - avoid_deadlock

# Return JSON only, with this format:
# {{
#   "subgoals": [
#     {{
#       "type": "clear_path",
#       "object": [row, col],
#       "priority": 1
#     }},
#     {{
#       "type": "deliver_box",
#       "object": [row, col],
#       "target": [row, col],
#       "priority": 2
#     }}
#   ]
# }}

# Current state:
# {json.dumps(data["map_state"], ensure_ascii=False)}

# Text map:
# {data["text_map"]}
# """
#     return prompt.strip()



# def build_phase_recognition_prompt(task_instance):
#     data = task_instance.input_data
#     prompt = f"""
# You are solving a Sokoban planning task.

# Given the current Sokoban state, identify the current planning phase.

# Allowed phases:
# {json.dumps(data["allowed_phases"], ensure_ascii=False)}

# Return JSON only, with this format:
# {{
#   "phase": "clear_path",
#   "reason": "optional short explanation"
# }}

# Current state:
# {json.dumps(data["map_state"], ensure_ascii=False)}

# Text map:
# {data["text_map"]}
# """
#     return prompt.strip()


# def build_subproblem_ordering_prompt(task_instance):
#     data = task_instance.input_data
#     prompt = f"""
# You are solving a Sokoban planning task.

# Given multiple candidate subgoals, decide a better execution order.

# Return JSON only, with this format:
# {{
#   "ordered_subgoals": ["sg2", "sg1", "sg3"],
#   "reason": "optional short explanation"
# }}

# Candidate subgoals:
# {json.dumps(data["candidate_subgoals"], ensure_ascii=False)}
# """
#     return prompt.strip()


# def build_horizon_choice_prompt(task_instance):
#     data = task_instance.input_data
#     prompt = f"""
# You are solving a Sokoban planning task.

# Choose the better option between A and B.
# One option may be easier in the short term, while the other may be safer or better in the long term.

# Return JSON only, with this format:
# {{
#   "better_choice": "A",
#   "reason": "optional short explanation"
# }}

# Option A:
# {json.dumps(data["option_A"], ensure_ascii=False)}

# Option B:
# {json.dumps(data["option_B"], ensure_ascii=False)}
# """
#     return prompt.strip()