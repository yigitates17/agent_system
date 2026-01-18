import httpx
import warnings
from core.schemas import LLMConfig, LLMResponse, ToolCall
from core.base_tool import BaseTool
from providers.base_provider import BaseLLMProvider
from typing import Literal


class OllamaProvider(BaseLLMProvider):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"

    async def call(
        self,
        messages: list[dict[str, str]],
        tools: list[BaseTool] | None = None,
        response_format: Literal["text", "json"] | None = None,
    ) -> LLMResponse:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }

        if tools:
            warnings.warn(
                f"Tool calling with Ollama is model-dependent. "
                f"Model '{self.config.model}' may not support tools.",
                UserWarning,
            )
            payload["tools"] = self.format_tools(tools)

        fmt = response_format or self.config.response_format
        if fmt == "json":
            payload["format"] = "json"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()

        return self._parse_response(data)

    def format_tools(self, tools: list[BaseTool]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.get_input_schema(),
                },
            }
            for tool in tools
        ]

    def _parse_response(self, data: dict) -> LLMResponse:
        message = data.get("message", {})
        tool_calls = []

        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                tool_calls.append(
                    ToolCall(
                        tool_name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    )
                )

        return LLMResponse(
            content=message.get("content"),
            tool_calls=tool_calls,
            finish_reason="tool_use" if tool_calls else "stop",
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
            raw=data,
        )