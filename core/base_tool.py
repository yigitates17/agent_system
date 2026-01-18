from abc import ABC, abstractmethod
from pydantic import BaseModel
from core.schemas import ToolResult, ExecutionContext, LLMResponse

class BaseTool(ABC):
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel] | None = None  # optional, for validation/docs

    def get_input_schema(self) -> dict:
        return self.input_model.model_json_schema()

    def get_output_schema(self) -> dict | None:
        return self.output_model.model_json_schema() if self.output_model else None

    @abstractmethod
    async def execute(self, input: BaseModel, context: ExecutionContext) -> ToolResult:
        pass