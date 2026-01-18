# test_agent_runner.py

import asyncio
from dotenv import load_dotenv
load_dotenv()

from core.schemas import LLMConfig, ExecutionContext, AgentState
from core.base_agent import BaseAgent, AgentConfig
from providers.openrouter import OpenRouterProvider
from tools.infrastructure.current_time import CurrentTimeTool
from executors.agent_runner import AgentRunner


async def main():
    provider = OpenRouterProvider(
        LLMConfig(provider="openrouter", model="mistralai/devstral-2512:free")
    )
    
    agent = BaseAgent(
        config=AgentConfig(
            name="test_agent",
            system_prompt="You are a helpful assistant. Use tools when needed.",
            tools=[CurrentTimeTool()],
        ),
        provider=provider,
    )
    
    context = ExecutionContext(
        agent_state=agent.state,
        session_id="test-session",
    )
    
    runner = AgentRunner(agent, context)
    result = await runner.run("What time is it right now?")
    
    print("Final result:", result)
    print("Attempts:", len(agent.state.attempts))


asyncio.run(main())