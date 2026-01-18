import os
from anthropic import AsyncAnthropic
from core.schemas import LLMConfig, LLMResponse, ToolCall
from core.base_tool import BaseTool
from providers.base_provider import BaseLLMProvider
from typing import Literal


class AnthropicProvider(BaseLLMProvider):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = AsyncAnthropic(
            api_key=config.api_key or os.getenv("ANTHROPIC_API_KEY")
        )

    async def call(
        self,
        messages: list[dict[str, str]],
        tools: list[BaseTool] | None = None,
        response_format: Literal["text", "json"] | None = None,
    ) -> LLMResponse:
        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": messages,
        }

        if tools:
            kwargs["tools"] = self.format_tools(tools)

        response = await self.client.messages.create(**kwargs)

        return self._parse_response(response)

    def format_tools(self, tools: list[BaseTool]) -> list[dict]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.get_input_schema(),
            }
            for tool in tools
        ]

    def _parse_response(self, response) -> LLMResponse:
        content = None
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(tool_name=block.name, arguments=block.input)
                )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="tool_use" if tool_calls else "stop",
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            },
            raw=response,
        )