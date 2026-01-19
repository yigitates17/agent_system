# Corvus

A modular agent framework built from scratch as a learning project in software architecture and system design.

## What This Is

This isn't another LangChain wrapper. It's a ground-up implementation to understand how agent systems actually work — from base abstractions to production-ready custom agents.

**Why build from scratch?**

- Data scientist learning backend/system design
- Understand every component, not just glue libraries together
- Build portfolio piece that demonstrates architectural thinking
- Create foundation for real corporate workflow automation

## Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                        Executors                            │
│  ┌─────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ AgentRunner │  │ WorkflowExecutor │  │ GraphExecutor │  │
│  │   (smart)   │  │    (explicit)    │  │  (LangGraph)  │  │
│  └─────────────┘  └──────────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        BaseAgent                            │
│            (config + provider + state + tools)              │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    ┌──────────┐        ┌──────────┐        ┌──────────┐
    │ Provider │        │  Tools   │        │  State   │
    │ Anthropic│        │ Search   │        │ Memory   │
    │ OpenAI   │        │ PDF      │        │ Snapshot │
    │ Ollama   │        │ Custom   │        │ Store    │
    │OpenRouter│        │          │        │          │
    └──────────┘        └──────────┘        └──────────┘
```

## Execution Modes

| Mode | Executor | Who Decides | Use Case |
|------|----------|-------------|----------|
| Smart | `AgentRunner` | LLM picks tools | Exploration, chat |
| Explicit | `WorkflowExecutor` | You define steps | Predictable pipelines |
| Graph | `GraphExecutor` | Conditional edges | Complex flows with cycles |

## Features

- **Multi-provider support** — Anthropic, OpenAI, Ollama, OpenRouter
- **Tool abstraction** — Pydantic validation, schema generation
- **Checkpoints** — Human-in-the-loop approval, revision, go-back
- **State management** — Snapshots for rollback, session persistence
- **Registries** — Central tool and agent management
- **Error handling** — Retry strategies, graceful degradation

## Quick Start
```python
from core.base_agent import BaseAgent, AgentConfig
from core.schemas import LLMConfig, ExecutionContext, AgentState
from providers.openrouter import OpenRouterProvider
from executors.agent_runner import AgentRunner
from tools.infrastructure.duckduckgo_search import DuckDuckGoSearchTool

# 1. Provider
provider = OpenRouterProvider(
    LLMConfig(provider="openrouter", model="google/gemma-3-27b-it:free")
)

# 2. Agent
agent = BaseAgent(
    config=AgentConfig(
        name="research_agent",
        system_prompt="You are a research assistant.",
        tools=[DuckDuckGoSearchTool()],
    ),
    provider=provider,
)

# 3. Run
runner = AgentRunner(agent, ExecutionContext(agent_state=agent.state, session_id="1"))
result = await runner.run("Search for Python history")
```

## Workflow with Checkpoints
```python
from executors.workflow_executor import WorkflowExecutor
from checkpoints.user_approval import ConsoleApprovalHandler
from core.schemas import WorkflowDefinition, WorkflowStep

workflow = WorkflowDefinition(
    name="research_report",
    description="Research and create PDF",
    steps=[
        WorkflowStep(name="search", tool_name="web_search", prompt="...", checkpoint=True),
        WorkflowStep(name="outline", tool_name=None, prompt="Create outline..."),
        WorkflowStep(name="create_pdf", tool_name="pdf_creator", prompt="..."),
    ]
)

executor = WorkflowExecutor(agent, context, ConsoleApprovalHandler())
results = await executor.run(workflow)
```

## Project Structure
```
agent_system/
├── core/               # Base abstractions
│   ├── schemas.py      # All Pydantic models
│   ├── base_tool.py    # Tool contract
│   └── base_agent.py   # Agent config + state
├── providers/          # LLM integrations
├── executors/          # Execution strategies
├── tools/              # Tool implementations
│   ├── infrastructure/ # Generic (search, time)
│   └── domain/         # Business-specific
├── checkpoints/        # Human-in-the-loop
├── memory/             # State persistence
├── registries/         # Tool/agent catalogs
├── workflows/          # Workflow definitions
├── infra/              # Error handling, logging
└── utils/              # Helpers
```

## Design Principles

1. **Classes over functions** — Enforced contracts via abstract base classes
2. **Configuration over inheritance** — Agents differ by config, not subclasses
3. **Explicit over implicit** — No magic, readable flow
4. **Async by default** — Built for I/O-bound operations

## Adding Components

**New provider:**
```python
class NewProvider(BaseLLMProvider):
    async def call(self, messages, tools=None, response_format=None) -> LLMResponse:
        pass
    def format_tools(self, tools) -> list[dict]:
        pass
```

**New tool:**
```python
class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something"
    input_model = MyInput
    
    async def execute(self, input, context) -> ToolResult:
        pass
```

## Roadmap

- [x] Multi-provider support
- [x] Smart execution (AgentRunner)
- [x] Explicit execution (WorkflowExecutor)
- [x] Checkpoints
- [x] State snapshots
- [x] Tool/agent registries
- [ ] LangGraph integration (GraphExecutor)
- [ ] Streaming responses
- [ ] MCP support
- [ ] SQLite/Postgres persistence
- [ ] Usage tracking
- [ ] Context window management

## Built With

- Python 3.13
- Pydantic
- httpx
- anthropic / openai SDKs
- ddgs (DuckDuckGo search)
- fpdf2 (PDF generation)

---

*Built from scratch to understand how agent systems actually work.*