# agents/research_agent.py

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

# 1. Provider
provider = OpenRouterProvider(
    LLMConfig(
        provider="openrouter",
        model="mistralai/devstral-2512:free"
    )
)

# 2. Tools
tools = [
    DuckDuckGoSearchTool(),
    PDFCreatorTool(),
]

# 3. Agent config
config = AgentConfig(
    name="research_agent",
    system_prompt="""You are a research assistant. 
    Follow instructions carefully and return JSON when asked.""",
    tools=tools,
    max_iterations=10,
)

# 4. Agent
agent = BaseAgent(config=config, provider=provider)

# 5. Context
context = ExecutionContext(
    agent_state=agent.state,
    session_id="research-session-1",
)

# 6. Checkpoint handler (this is where you approve/revise)
checkpoint_handler = ConsoleApprovalHandler()

# 7. Workflow executor (NOT AgentRunner)
executor = WorkflowExecutor(
    agent=agent,
    context=context,
    checkpoint_handler=checkpoint_handler,
)


# 8. Run
async def main():
    results = await executor.run(ResearchWorkflow)
    print("\n=== DONE ===")
    for step_name, result in results.items():
        print(f"{step_name}: {result.success}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())