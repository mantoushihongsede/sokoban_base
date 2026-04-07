import json
from typing import Any, Dict, List, Optional

from analyzers.tool_dispatch import dispatch_tool_call
from analyzers.tools import SokobanAnalysisTools


class FunctionCallRunner:
    def __init__(
        self,
        client,
        model_name: str,
        tools: SokobanAnalysisTools,
        openai_tools: List[Dict[str, Any]],
        max_rounds: int = 12,
        max_tool_calls: Optional[int] = None,
    ):
        self.client = client
        self.model_name = model_name
        self.tools = tools
        self.openai_tools = openai_tools
        self.max_rounds = max_rounds
        self.max_tool_calls = max_tool_calls

    def _parse_function_calls(self, response) -> List[Any]:
        calls = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "function_call":
                calls.append(item)
        return calls

    def _build_function_call_output_items(
        self,
        function_calls: List[Any],
    ) -> List[Dict[str, Any]]:
        tool_outputs = []

        for call in function_calls:
            tool_name = call.name
            raw_args = call.arguments

            try:
                arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception as e:
                result = {
                    "ok": False,
                    "error": f"Invalid JSON arguments: {e}",
                    "raw_arguments": raw_args,
                }
            else:
                try:
                    result = dispatch_tool_call(
                        tools=self.tools,
                        tool_name=tool_name,
                        arguments=arguments,
                    )
                except Exception as e:
                    result = {
                        "ok": False,
                        "error": f"Tool execution failed: {type(e).__name__}: {e}",
                        "tool_name": tool_name,
                    }

            tool_outputs.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result, ensure_ascii=False),
            })

        return tool_outputs

    def _build_initial_input_items(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> List[Dict[str, Any]]:
        # 用尽量简单的结构，避免 provider 对 content item 挑剔
        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

    def _finalize_without_tools(
        self,
        conversation_items: List[Dict[str, Any]],
        logs: List[Dict[str, Any]],
        tool_call_count: int,
        reason: str,
    ) -> Dict[str, Any]:
        final_items = list(conversation_items)
        final_items.append({
            "role": "user",
            "content": (
                "Tool use must stop now. "
                "Do not call any more tools. "
                "Return your final answer now in the required JSON format only."
            ),
        })

        response = self.client.create_response(
            model=self.model_name,
            input_items=final_items,
            tools=[],
        )

        final_text = self.client.extract_text(response)

        logs.append({
            "round": "finalize",
            "response_id": getattr(response, "id", None),
            "output_text": final_text,
            "tool_call_count": tool_call_count,
            "finalize_reason": reason,
        })

        return {
            "final_text": final_text,
            "response": response,
            "logs": logs,
            "tool_call_count": tool_call_count,
            "error": reason,
        }

    def run(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> Dict[str, Any]:
        logs: List[Dict[str, Any]] = []
        tool_call_count = 0

        # 本地维护本轮 function-calling 上下文
        conversation_items: List[Dict[str, Any]] = self._build_initial_input_items(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        response = self.client.create_response(
            model=self.model_name,
            input_items=conversation_items,
            tools=self.openai_tools,
        )

        logs.append({
            "round": 0,
            "response_id": getattr(response, "id", None),
            "output_text": self.client.extract_text(response),
            "tool_call_count": tool_call_count,
        })

        for round_idx in range(1, self.max_rounds + 1):
            function_calls = self._parse_function_calls(response)

            if not function_calls:
                return {
                    "final_text": self.client.extract_text(response),
                    "response": response,
                    "logs": logs,
                    "tool_call_count": tool_call_count,
                }

            requested_calls = len(function_calls)

            if (
                self.max_tool_calls is not None
                and tool_call_count + requested_calls > self.max_tool_calls
            ):
                logs.append({
                    "round": round_idx,
                    "budget_stop": True,
                    "message": (
                        f"Tool call budget exceeded: "
                        f"current={tool_call_count}, "
                        f"requested={requested_calls}, "
                        f"max_tool_calls={self.max_tool_calls}"
                    ),
                    "pending_function_calls": [
                        {
                            "name": fc.name,
                            "arguments": fc.arguments,
                            "call_id": fc.call_id,
                        }
                        for fc in function_calls
                    ],
                    "tool_call_count": tool_call_count,
                })

                return self._finalize_without_tools(
                    conversation_items=conversation_items,
                    logs=logs,
                    tool_call_count=tool_call_count,
                    reason=f"Exceeded max_tool_calls={self.max_tool_calls}",
                )

            # 把 assistant 的 function_call 写回上下文
            for call in function_calls:
                conversation_items.append({
                    "type": "function_call",
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": call.arguments if isinstance(call.arguments, str) else json.dumps(call.arguments, ensure_ascii=False),
                })

            tool_outputs = self._build_function_call_output_items(function_calls)
            tool_call_count += requested_calls

            # 再把工具结果写回上下文
            conversation_items.extend(tool_outputs)

            logs.append({
                "round": round_idx,
                "function_calls": [
                    {
                        "name": fc.name,
                        "arguments": fc.arguments,
                        "call_id": fc.call_id,
                    }
                    for fc in function_calls
                ],
                "tool_outputs": tool_outputs,
                "tool_call_count": tool_call_count,
            })

            response = self.client.create_response(
                model=self.model_name,
                input_items=conversation_items,
                tools=self.openai_tools,
            )

            logs.append({
                "round": round_idx,
                "response_id": getattr(response, "id", None),
                "output_text": self.client.extract_text(response),
                "tool_call_count": tool_call_count,
            })

        return self._finalize_without_tools(
            conversation_items=conversation_items,
            logs=logs,
            tool_call_count=tool_call_count,
            reason=f"Exceeded max_rounds={self.max_rounds}",
        )