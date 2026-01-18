# checkpoints/base_checkpoint.py

from abc import ABC, abstractmethod
from core.schemas import CheckpointResponse, ToolResult


class BaseCheckpointHandler(ABC):
    @abstractmethod
    async def handle(self, step_name: str, result: ToolResult) -> CheckpointResponse:
        pass