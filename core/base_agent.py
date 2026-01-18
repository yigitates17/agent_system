# core/base_agent.py

from pydantic import BaseModel, Field
from core.schemas import AgentState, ExecutionContext, LLMResponse
from core.base_tool import BaseTool
from providers.base_provider import BaseLLMProvider
from typing import Literal

class AgentConfig(BaseModel):
    name: str
    system_prompt: str
    tools: list[BaseTool] = Field(default_factory=list)
    max_iterations: int = 10
    response_format: Literal["text", "json"] = "text"


    class Config:
        arbitrary_types_allowed = True


class BaseAgent:
    def __init__(self, config: AgentConfig, provider: BaseLLMProvider):
        self.config = config
        self.provider = provider
        self.state = AgentState()

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def tools(self) -> list[BaseTool]:
        return self.config.tools

    def get_tool(self, name: str) -> BaseTool | None:
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def build_messages(self) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.config.system_prompt}]
        messages.extend(self.state.chat_history)
        return messages

    async def call_llm(self) -> LLMResponse:
        messages = self.build_messages()
        return await self.provider.call(
            messages=messages,
            tools=self.tools if self.tools else None,
            response_format=self.config.response_format,
        )

    def add_user_message(self, content: str):
        self.state.chat_history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        self.state.chat_history.append({"role": "assistant", "content": content})

    def add_tool_result(self, tool_name: str, result: str):
        self.state.chat_history.append({
            "role": "user",  # tool results go back as user message
            "content": f"[Tool: {tool_name}] Result: {result}",
        })

    def reset(self):
        self.state = AgentState()