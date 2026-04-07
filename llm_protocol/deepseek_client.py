from openai import OpenAI


class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def chat(self, system_prompt: str, user_prompt: str, model: str = "deepseek-chat", temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content