from openai import OpenAI
from pydantic import BaseModel
from typing import Any, List, Tuple

class BoxPriorityOutput(BaseModel):
    box_priority_order: List[Tuple[int, int]]

class AssignmentItem(BaseModel):
    box: Tuple[int, int]
    target: Tuple[int, int]

class BoxAssignmentOutput(BaseModel):
    assignments: List[AssignmentItem]


class OpenAIClient:
    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def create_response(
        self,
        *,
        model: str,
        input_items: Any,
        tools: Any = None,
    ):
        kwargs = {
            "model": model,
            "input": input_items,
        }
        if tools is not None:
            kwargs["tools"] = tools
        return self.client.responses.create(**kwargs)

    def extract_text(self, response) -> str:
        return getattr(response, "output_text", "") or ""

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "gpt-4.1-mini",
        # temperature: float = 0.0,
    ) -> str:
        response = self.client.responses.create(
            model=model,
            instructions=system_prompt,
            input=user_prompt,
            # text_format=BoxPriorityOutput,
        )
        return self.extract_text(response)
