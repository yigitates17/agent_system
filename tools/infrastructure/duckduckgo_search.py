from ddgs import DDGS
from datetime import datetime
from pydantic import BaseModel
from core.base_tool import BaseTool
from core.schemas import ToolResult, ExecutionContext

class DuckSearchInput(BaseModel):
    search_key: str
    max_results: int = 5

class DuckDuckGoSearchTool(BaseTool):
    name = "duckduckgo_web_search"
    description = "Searches the web using DuckDuckGo"
    input_model = DuckSearchInput

    async def execute(self, input: DuckSearchInput, context: ExecutionContext) -> ToolResult:
        try:
            results = DDGS().text(input.search_key, max_results=input.max_results)
            return ToolResult(
                success=True,
                tool_name=self.name,
                input=input.model_dump(),
                data={"results": results},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                input=input.model_dump(),
                error=str(e),
            )