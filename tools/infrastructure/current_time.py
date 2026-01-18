# tools/infrastructure/current_time.py

from datetime import datetime
from pydantic import BaseModel
from core.base_tool import BaseTool
from core.schemas import ToolResult, ExecutionContext


class CurrentTimeInput(BaseModel):
    timezone: str = "GMT"


class CurrentTimeTool(BaseTool):
    name = "current_time"
    description = "Returns the current date and time"
    input_model = CurrentTimeInput

    async def execute(self, input: CurrentTimeInput, context: ExecutionContext) -> ToolResult:
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(input.timezone)
            now = datetime.now(tz)
            
            return ToolResult(
                success=True,
                tool_name=self.name,
                input=input.model_dump(),
                data={"datetime": now.isoformat(), "timezone": input.timezone},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                input=input.model_dump(),
                error=str(e),
            )