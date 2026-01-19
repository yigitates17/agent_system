from pydantic import BaseModel, Field
from typing import Any, Literal
from datetime import datetime
from enum import Enum

class ToolResult(BaseModel):
    success: bool
    data: Any | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)  # execution time, tokens used, etc.

class LLMConfig(BaseModel):
    provider: Literal["anthropic", "openai", "ollama", "openrouter"]
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    response_format: Literal["text", "json"] = "text"
    base_url: str | None = None  # for ollama/openrouter
    api_key: str | None = None   # loaded from env if None

class ToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any]

class LLMResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: Literal["stop", "tool_use", "length", "error"]
    usage: dict[str, int] = Field(default_factory=dict)  # prompt_tokens, completion_tokens
    raw: Any | None = None  # original provider response for debugging

class CheckpointDecision(str, Enum):
    APPROVE = "approve"
    REVISE = "revise"
    GO_BACK = "go_back"
    STOP = "stop"

class CheckpointResponse(BaseModel):
    decision: CheckpointDecision
    feedback: str | None = None
    go_back_to: str | None = None  # step name, required if decision is GO_BACK

class Attempt(BaseModel):
    step: str
    attempt_number: int
    timestamp: datetime = Field(default_factory=datetime.now)
    llm_response: LLMResponse | None = None
    tool_results: list[ToolResult] = Field(default_factory=list)
    checkpoint_response: CheckpointResponse | None = None
    error: str | None = None

class AgentState(BaseModel):
    current_step: str | None = None
    current_attempt: int = 1
    outputs: dict[str, Any] = Field(default_factory=dict)       # step_name -> result
    attempts: list[Attempt] = Field(default_factory=list)       # full history
    chat_history: list[dict[str, str]] = Field(default_factory=list)  # role, content
    feedback: str | None = None  # current revision feedback
    snapshots: dict[str, "AgentState"] = Field(default_factory=dict)  # for go_back

class ExecutionContext(BaseModel):
    agent_state: AgentState
    user_id: str | None = None
    session_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)  # db connections, feature flags, etc.

    class Config:
        arbitrary_types_allowed = True  # for db connections etc.

class WorkflowStep(BaseModel):
    name: str
    tool_name: str | None = None  # Allow None for LLM-only steps
    prompt: str
    checkpoint: bool = False
    input_override: dict[str, Any] | None = None


class WorkflowDefinition(BaseModel):
    name: str
    description: str
    steps: list[WorkflowStep]