from pydantic import BaseModel
from typing import List, Tuple
from anthropic import Anthropic, transform_schema


class BoxPriorityOutput(BaseModel):
    box_priority_order: List[Tuple[int, int]]

class AssignmentItem(BaseModel):
    box: Tuple[int, int]
    target: Tuple[int, int]

class BoxAssignmentOutput(BaseModel):
    assignments: List[AssignmentItem]



class ClaudeClient:
    def __init__(self, api_key: str, base_url: str):
        self.client = Anthropic(api_key=api_key, base_url=base_url)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "claude-3-5-sonnet-latest",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        response = self.client.messages.create(
            model=model,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            # output_config={
            #     "effort": "medium", #限制思考深度。看情况。没有像gemini一样思考个不停。
            #     "format": {
            #         "type": "json_schema",
            #         "schema": transform_schema(BoxPriorityOutput),
            #     }
            # }
        )
        return response.content[0].text