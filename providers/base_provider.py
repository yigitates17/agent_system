from abc import ABC, abstractmethod
from pydantic import BaseModel
from core.base_tool import BaseTool, ExecutionContext
from core.schemas import LLMResponse, LLMConfig
from typing import Literal

class BaseLLMProvider(ABC):
    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    async def call(
        self,
        messages: list[dict[str, str]],
        tools: list[BaseTool] | None = None,
        response_format: Literal["text", "json"] | None = None,
    ) -> LLMResponse:
        pass

    @abstractmethod
    def format_tools(self, tools: list[BaseTool]) -> list[dict]:
        pass