# infra/error_handler.py

from enum import Enum
from pydantic import BaseModel
import asyncio


class ErrorType(str, Enum):
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    INVALID_RESPONSE = "invalid_response"
    TOOL_EXECUTION = "tool_execution"
    API_ERROR = "api_error"
    UNKNOWN = "unknown"


class ErrorStrategy(BaseModel):
    error_type: ErrorType
    max_retries: int = 3
    wait_seconds: float = 1.0
    exponential_backoff: bool = True


DEFAULT_STRATEGIES = {
    ErrorType.RATE_LIMIT: ErrorStrategy(
        error_type=ErrorType.RATE_LIMIT,
        max_retries=5,
        wait_seconds=5.0,
        exponential_backoff=True,
    ),
    ErrorType.TIMEOUT: ErrorStrategy(
        error_type=ErrorType.TIMEOUT,
        max_retries=3,
        wait_seconds=2.0,
        exponential_backoff=False,
    ),
    ErrorType.INVALID_RESPONSE: ErrorStrategy(
        error_type=ErrorType.INVALID_RESPONSE,
        max_retries=3,
        wait_seconds=0.0,
        exponential_backoff=False,
    ),
    ErrorType.TOOL_EXECUTION: ErrorStrategy(
        error_type=ErrorType.TOOL_EXECUTION,
        max_retries=2,
        wait_seconds=0.0,
        exponential_backoff=False,
    ),
    ErrorType.API_ERROR: ErrorStrategy(
        error_type=ErrorType.API_ERROR,
        max_retries=3,
        wait_seconds=1.0,
        exponential_backoff=True,
    ),
}


class ErrorHandler:
    def __init__(self, strategies: dict[ErrorType, ErrorStrategy] | None = None):
        self.strategies = strategies or DEFAULT_STRATEGIES
        self.attempt_counts: dict[str, int] = {}

    def classify_error(self, error: Exception) -> ErrorType:
        error_str = str(error).lower()
        
        if "rate" in error_str or "429" in error_str:
            return ErrorType.RATE_LIMIT
        elif "timeout" in error_str:
            return ErrorType.TIMEOUT
        elif "invalid" in error_str or "validation" in error_str:
            return ErrorType.INVALID_RESPONSE
        elif "401" in error_str or "403" in error_str or "500" in error_str:
            return ErrorType.API_ERROR
        
        return ErrorType.UNKNOWN

    async def handle(
        self, 
        error: Exception, 
        context_key: str,
    ) -> tuple[bool, float]:
        """
        Returns (should_retry, wait_seconds).
        """
        error_type = self.classify_error(error)
        strategy = self.strategies.get(error_type, DEFAULT_STRATEGIES[ErrorType.UNKNOWN])
        
        # Track attempts
        self.attempt_counts[context_key] = self.attempt_counts.get(context_key, 0) + 1
        current_attempt = self.attempt_counts[context_key]
        
        if current_attempt >= strategy.max_retries:
            return False, 0.0
        
        # Calculate wait time
        wait = strategy.wait_seconds
        if strategy.exponential_backoff:
            wait = strategy.wait_seconds * (2 ** (current_attempt - 1))
        
        return True, wait

    def reset(self, context_key: str) -> None:
        self.attempt_counts.pop(context_key, None)

    def reset_all(self) -> None:
        self.attempt_counts = {}