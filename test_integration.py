# test_integration.py

import asyncio
from core.schemas import LLMConfig, ExecutionContext, AgentState
from providers.anthropic import AnthropicProvider
from providers.openrouter import OpenRouterProvider
from tools.infrastructure.current_time import CurrentTimeTool, CurrentTimeInput

import os
from dotenv import load_dotenv
load_dotenv()

print("API Key loaded:", os.getenv("OPENROUTER_API_KEY")[:10], "...")  # prints first 10 chars

async def main():
    # Setup
    config = LLMConfig(provider="openrouter", model="google/gemma-3-27b-it:free")
    provider = OpenRouterProvider(config)
    tool = CurrentTimeTool()

    # Test 1: Tool schema generation
    print("Schema:", tool.get_input_schema())

    # Test 2: Tool execution
    context = ExecutionContext(
        agent_state=AgentState(),
        session_id="test-123",
    )
    result = await tool.execute(CurrentTimeInput(timezone="GMT"), context)
    print("Tool result:", result)

    # Test 3: LLM call with tool
    response = await provider.call(
        messages=[{"role": "user", "content": "What time is it?"}],
        tools=[tool],
    )
    print("LLM response:", response)

    # Test 4: If LLM requested tool, execute it
    if response.tool_calls:
        tc = response.tool_calls[0]
        print(f"LLM wants to call: {tc.tool_name} with {tc.arguments}")
        
        validated_input = tool.input_model(**tc.arguments)
        result = await tool.execute(validated_input, context)
        print("Execution result:", result)


asyncio.run(main())