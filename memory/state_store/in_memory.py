# memory/state_store/in_memory.py

from core.schemas import AgentState
from memory.state_store.base_store import BaseStateStore


class InMemoryStateStore(BaseStateStore):
    def __init__(self):
        self._store: dict[str, AgentState] = {}
    
    async def save(self, session_id: str, state: AgentState) -> None:
        self._store[session_id] = state.model_copy(deep=True)
    
    async def load(self, session_id: str) -> AgentState | None:
        state = self._store.get(session_id)
        return state.model_copy(deep=True) if state else None
    
    async def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)
    
    async def exists(self, session_id: str) -> bool:
        return session_id in self._store