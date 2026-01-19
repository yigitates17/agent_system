# executors/workflow_executor.py

from core.base_agent import BaseAgent
from core.schemas import (
    ExecutionContext, 
    ToolResult, 
    Attempt,
    WorkflowDefinition,
    CheckpointResponse,
    CheckpointDecision,
)
from pydantic import ValidationError
from datetime import datetime
from typing import Callable, Awaitable
import json
import re


class WorkflowExecutor:
    def __init__(
        self, 
        agent: BaseAgent, 
        context: ExecutionContext,
        checkpoint_handler: Callable[[str, ToolResult], Awaitable[CheckpointResponse]] | None = None,
    ):
        self.agent = agent
        self.context = context
        self.checkpoint_handler = checkpoint_handler

    async def run(self, workflow: WorkflowDefinition) -> dict[str, ToolResult]:
        """Execute workflow steps in order, respecting checkpoints."""
        results: dict[str, ToolResult] = {}
        
        step_index = 0
        while step_index < len(workflow.steps):
            step = workflow.steps[step_index]
            self.agent.state.current_step = step.name
            
            # Execute step
            result = await self._execute_step(step)
            results[step.name] = result

            if result.success and result.data:
                self.agent.state.chat_history.append({
                    "role": "user",
                    "content": f"[Step: {step.name}] Result:\n{result.data}"
                })
            
            # Log attempt
            attempt = Attempt(
                step=step.name,
                attempt_number=self.agent.state.current_attempt,
                timestamp=datetime.now(),
                tool_results=[result],
            )
            self.agent.state.attempts.append(attempt)
            self.agent.state.outputs[step.name] = result.data
            
            # Handle checkpoint
            if step.checkpoint and self.checkpoint_handler:
                checkpoint_response = await self.checkpoint_handler.handle(step.name, result)
                attempt.checkpoint_response = checkpoint_response
                
                next_index = self._handle_checkpoint(
                    checkpoint_response, 
                    step_index, 
                    workflow
                )
                
                if next_index == -1:  # stop
                    break
                    
                step_index = next_index
                continue
            
            step_index += 1
        
        return results

    def _extract_json(self, text: str) -> dict | None:
        """Extract JSON from LLM response, handling markdown code blocks."""
        if not text:
            return None
        
        # Remove markdown code blocks
        pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        match = re.search(pattern, text)
        
        if match:
            text = match.group(1)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    async def _execute_llm_step(self, step) -> ToolResult:
        """Step where LLM thinks and responds, no tool."""
        messages = self.agent.build_messages()
        messages.append({
            "role": "user",
            "content": step.prompt,
        })
        
        response = await self.agent.provider.call(
            messages=messages,
            tools=None,
            response_format="json",  # Ask for JSON
        )
        
        if response.content:
            self.agent.add_assistant_message(response.content)

        parsed = self._extract_json(response.content)

        self.agent.state.outputs[step.name] = parsed or {"response": response.content}
        
        return ToolResult(
            success=True,
            tool_name="llm_response",
            input={"prompt": step.prompt},
            data=parsed or {"response": response.content},
        )

    async def _execute_step(self, step) -> ToolResult:

        if step.tool_name is None:
            return await self._execute_llm_step(step)
        
        tool = self.agent.get_tool(step.tool_name)
        
        if not tool:
            return ToolResult(
                success=False,
                tool_name=step.tool_name,
                input={},
                error=f"Tool '{step.tool_name}' not found",
            )
        
        # Get input: either from override, LLM, or empty
        if step.input_override:
            tool_input = step.input_override
        else:
            tool_input = await self._get_input_from_llm(step, tool)
        
        # Validate
        try:
            validated_input = tool.input_model(**tool_input)
        except ValidationError as e:
            return ToolResult(
                success=False,
                tool_name=step.tool_name,
                input=tool_input,
                error=f"Invalid input: {e}",
            )
        
        # Execute
        try:
            return await tool.execute(validated_input, self.context)
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=step.tool_name,
                input=tool_input,
                error=f"Execution error: {e}",
            )

    async def _get_input_from_llm(self, step, tool) -> dict:
        """Ask LLM to provide tool input based on step prompt."""
        messages = self.agent.build_messages()
        messages.append({
            "role": "user",
            "content": f"{step.prompt}\n\nUse the {tool.name} tool to complete this.",
        })
        
        response = await self.agent.provider.call(
            messages=messages,
            tools=[tool],
        )
        
        if response.tool_calls:
            return response.tool_calls[0].arguments
        
        return {}

    def _handle_checkpoint(
        self, 
        response: CheckpointResponse, 
        current_index: int,
        workflow: WorkflowDefinition,
    ) -> int:
        """Return next step index, or -1 to stop."""
        
        if response.decision == CheckpointDecision.APPROVE:
            return current_index + 1
        
        elif response.decision == CheckpointDecision.STOP:
            return -1
        
        elif response.decision == CheckpointDecision.REVISE:
            self.agent.state.feedback = response.feedback
            self.agent.state.current_attempt += 1
            return current_index  # retry same step
        
        elif response.decision == CheckpointDecision.GO_BACK:
            # Find step index by name
            for i, s in enumerate(workflow.steps):
                if s.name == response.go_back_to:
                    self.agent.state.current_attempt = 1
                    return i
            return current_index + 1  # not found, continue
        
        return current_index + 1