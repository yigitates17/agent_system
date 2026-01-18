import os
import httpx
from core.schemas import LLMConfig, LLMResponse, ToolCall
from core.base_tool import BaseTool
from providers.base_provider import BaseLLMProvider
from typing import Literal


class OpenRouterProvider(BaseLLMProvider):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://openrouter.ai/api/v1"
        self.api_key = config.api_key or os.getenv("OPENROUTER_API_KEY")

    async def call(
        self,
        messages: list[dict[str, str]],
        tools: list[BaseTool] | None = None,
        response_format: Literal["text", "json"] | None = None,
    ) -> LLMResponse:
        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": messages,
        }

        if tools:
            payload["tools"] = self.format_tools(tools)

        fmt = response_format or self.config.response_format
        if fmt == "json":
            payload["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
            # print("API response:", data)
            return self._parse_response(data)

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

        if "error" in data:
            return LLMResponse(
                content=None,
                tool_calls=[],
                finish_reason="error",
                usage={},
                raw=data,
            )

        choice = data["choices"][0]
        message = choice["message"]
        tool_calls = []

        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                tool_calls.append(
                    ToolCall(
                        tool_name=tc["function"]["name"],
                        arguments=eval(tc["function"]["arguments"]),
                    )
                )

        return LLMResponse(
            content=message.get("content"),
            tool_calls=tool_calls,
            finish_reason="tool_use" if tool_calls else choice.get("finish_reason", "stop"),
            usage={
                "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
            },
            raw=data,
        )