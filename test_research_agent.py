# test_research_agent.py

import asyncio
from dotenv import load_dotenv
load_dotenv()

from core.base_agent import BaseAgent, AgentConfig
from core.schemas import LLMConfig, ExecutionContext, AgentState
from providers.openrouter import OpenRouterProvider
from tools.infrastructure.duckduckgo_search import DuckDuckGoSearchTool
from tools.domain.research.pdf_creator import PDFCreatorTool
from executors.workflow_executor import WorkflowExecutor
from workflows.research_workflow import ResearchWorkflow
from checkpoints.user_approval import ConsoleApprovalHandler


async def main():
    # Provider
    provider = OpenRouterProvider(
        LLMConfig(
            provider="openrouter",
            model="google/gemma-3-27b-it:free"
        )
    )

    # Agent
    agent = BaseAgent(
        config=AgentConfig(
            name="research_agent",
            system_prompt="""You are a research assistant. 
            Follow instructions carefully and return JSON when asked.""",
            tools=[DuckDuckGoSearchTool(), PDFCreatorTool()],
            max_iterations=10,
        ),
        provider=provider,
    )

    # Context
    context = ExecutionContext(
        agent_state=agent.state,
        session_id="research-session-1",
    )

    # Executor with checkpoints
    executor = WorkflowExecutor(
        agent=agent,
        context=context,
        checkpoint_handler=ConsoleApprovalHandler(),
    )

    # Run workflow
    print("Starting research workflow...")
    print("Topic: Python programming history\n")
    
    results = await executor.run(ResearchWorkflow)
    
    print("\n" + "="*50)
    print("WORKFLOW COMPLETE")
    print("="*50)
    for step_name, result in results.items():
        print(f"{step_name}: {'✓' if result.success else '✗'}")
        if result.data:
            print(f"  Data: {str(result.data)[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())