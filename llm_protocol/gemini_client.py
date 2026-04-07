from google import genai
from google.genai import types


class GeminiClient:
    def __init__(self, api_key: str, base_url: str):
        self.client = genai.Client(
        api_key=api_key,
        vertexai=True, # 优先使用vertexai协议访问，稳定性更高
        http_options={
            "base_url": base_url
        })

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "gemini-3.0-flash", #必须是3.0系列的模型才支持thinking_level参数
        temperature: float = 0.0,
        thinking_level = "low",
    ) -> str:
        response = self.client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                thinking_config=types.ThinkingConfig(thinking_level=thinking_level), #必须限制，否则模型是高思考，价格太贵。本身任务简单。不需要他复杂的思考过程。
                response_mime_type="application/json",  #这里可以要求的更严格一些，可以用response_json_schema但目前Gemini有按要求回复json。用不到。
            ),  
        )
        return response.text