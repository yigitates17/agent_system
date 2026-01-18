import os
from openai import AsyncOpenAI
from core.schemas import LLMConfig, LLMResponse, ToolCall
from core.base_tool import BaseTool
from providers.base_provider import BaseLLMProvider
from typing import Literal


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = AsyncOpenAI(
            api_key=config.api_key or os.getenv("OPENAI_API_KEY")
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

        fmt = response_format or self.config.response_format
        if fmt == "json":
            kwargs["response_format"] = {"type": "json_object"}

        response = await self.client.chat.completions.create(**kwargs)

        return self._parse_response(response)

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

    def _parse_response(self, response) -> LLMResponse:
        message = response.choices[0].message
        tool_calls = []

        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        tool_name=tc.function.name,
                        arguments=eval(tc.function.arguments),  # comes as string
                    )
                )

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason="tool_use" if tool_calls else response.choices[0].finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
            raw=response,
        )