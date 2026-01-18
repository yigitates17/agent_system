# tools/infrastructure/call_agent.py

from pydantic import BaseModel
from core.base_tool import BaseTool
from core.schemas import ToolResult, ExecutionContext
from registries.agent_registry import AgentRegistry
from executors.agent_runner import AgentRunner


class CallAgentInput(BaseModel):
    agent_name: str
    task: str


class CallAgentTool(BaseTool):
    name = "call_agent"
    description = "Delegates a task to another specialized agent"
    input_model = CallAgentInput

    async def execute(self, input: CallAgentInput, context: ExecutionContext) -> ToolResult:
        registry = AgentRegistry()
        agent = registry.get(input.agent_name)
        
        if not agent:
            available = registry.get_names()
            return ToolResult(
                success=False,
                tool_name=self.name,
                input=input.model_dump(),
                error=f"Agent '{input.agent_name}' not found. Available: {available}",
            )
        
        # Run the agent
        runner = AgentRunner(agent, context)
        result = await runner.run(input.task)
        
        return ToolResult(
            success=True,
            tool_name=self.name,
            input=input.model_dump(),
            data={"agent": input.agent_name, "result": result},
        )