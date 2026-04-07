import json


def parse_json_response(text: str):
    text = text.strip()

    # 有些模型会返回 ```json ... ```
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(text), None
    except Exception as e:
        return None, str(e)