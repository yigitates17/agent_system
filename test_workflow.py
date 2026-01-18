# test_workflow.py

import asyncio
from dotenv import load_dotenv
load_dotenv()

from core.schemas import (
    LLMConfig, ExecutionContext, AgentState,
    WorkflowDefinition, WorkflowStep,
    CheckpointResponse, CheckpointDecision,
)
from core.base_agent import BaseAgent, AgentConfig
from providers.openrouter import OpenRouterProvider
from tools.infrastructure.current_time import CurrentTimeTool
from executors.workflow_executor import WorkflowExecutor


async def simple_checkpoint_handler(step_name: str, result) -> CheckpointResponse:
    """Simple auto-approve for testing."""
    print(f"\n[Checkpoint] Step '{step_name}' completed.")
    print(f"Result: {result.data}")
    return CheckpointResponse(decision=CheckpointDecision.APPROVE)


async def main():
    provider = OpenRouterProvider(
        LLMConfig(provider="openrouter", model="mistralai/devstral-2512:free")
    )
    
    agent = BaseAgent(
        config=AgentConfig(
            name="workflow_agent",
            system_prompt="You are a helpful assistant.",
            tools=[CurrentTimeTool()],
        ),
        provider=provider,
    )
    
    context = ExecutionContext(
        agent_state=agent.state,
        session_id="test-workflow",
    )
    
    workflow = WorkflowDefinition(
        name="time_check",
        description="Get the current time",
        steps=[
            WorkflowStep(
                name="get_time",
                tool_name="current_time",
                prompt="Get the current time in GMT",
                checkpoint=True,
            ),
        ],
    )
    
    executor = WorkflowExecutor(
        agent=agent,
        context=context,
        checkpoint_handler=simple_checkpoint_handler,
    )
    
    results = await executor.run(workflow)
    print("\nFinal results:", results)


asyncio.run(main())