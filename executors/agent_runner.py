# executors/agent_runner.py

from core.base_agent import BaseAgent
from core.schemas import ExecutionContext, ToolResult, Attempt
from pydantic import ValidationError
from datetime import datetime


class AgentRunner:
    def __init__(self, agent: BaseAgent, context: ExecutionContext):
        self.agent = agent
        self.context = context

    async def run(self, task: str) -> str:
        """Run agent until completion or max iterations."""
        self.agent.add_user_message(task)
        
        for iteration in range(self.agent.config.max_iterations):
            response = await self.agent.call_llm()

            if response.finish_reason == "error":
                error_msg = response.raw.get("error", {}).get("message", "Unknown error")
                return f"API error: {error_msg}"
            
            # Log attempt
            attempt = Attempt(
                step=f"iteration_{iteration}",
                attempt_number=iteration + 1,
                timestamp=datetime.now(),
                llm_response=response,
            )
            self.agent.state.attempts.append(attempt)
            
            # No tool calls = done
            if not response.tool_calls:
                if response.content:
                    self.agent.add_assistant_message(response.content)
                return response.content or ""
            
            # Execute each tool call
            tool_results = await self._execute_tool_calls(response.tool_calls)
            attempt.tool_results = tool_results
            
            # Feed results back to LLM
            for tc, result in zip(response.tool_calls, tool_results):
                self._add_tool_result_message(tc.tool_name, result)
        
        return f"Max iterations ({self.agent.config.max_iterations}) reached"

    async def _execute_tool_calls(self, tool_calls: list) -> list[ToolResult]:
        results = []
        
        for tc in tool_calls:
            tool = self.agent.get_tool(tc.tool_name)
            
            if not tool:
                results.append(ToolResult(
                    success=False,
                    tool_name=tc.tool_name,
                    input=tc.arguments,
                    error=f"Tool '{tc.tool_name}' not found",
                ))
                continue
            
            # Validate input
            try:
                validated_input = tool.input_model(**tc.arguments)
            except ValidationError as e:
                results.append(ToolResult(
                    success=False,
                    tool_name=tc.tool_name,
                    input=tc.arguments,
                    error=f"Invalid input: {e}",
                ))
                continue
            
            # Execute
            try:
                result = await tool.execute(validated_input, self.context)
                results.append(result)
            except Exception as e:
                results.append(ToolResult(
                    success=False,
                    tool_name=tc.tool_name,
                    input=tc.arguments,
                    error=f"Execution error: {e}",
                ))
        
        return results

    def _add_tool_result_message(self, tool_name: str, result: ToolResult):
        if result.success:
            content = f"[Tool: {tool_name}] Result: {result.data}"
        else:
            content = f"[Tool: {tool_name}] Error: {result.error}"
        
        self.agent.state.chat_history.append({
            "role": "user",
            "content": content,
        })