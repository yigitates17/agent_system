# memory/state_store/base_store.py

from abc import ABC, abstractmethod
from core.schemas import AgentState


class BaseStateStore(ABC):
    @abstractmethod
    async def save(self, session_id: str, state: AgentState) -> None:
        pass
    
    @abstractmethod
    async def load(self, session_id: str) -> AgentState | None:
        pass
    
    @abstractmethod
    async def delete(self, session_id: str) -> None:
        pass
    
    @abstractmethod
    async def exists(self, session_id: str) -> bool:
        pass