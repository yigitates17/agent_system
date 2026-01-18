# memory/state_store/snapshot_manager.py

from core.schemas import AgentState
from memory.state_store.base_store import BaseStateStore


class SnapshotManager:
    def __init__(self, store: BaseStateStore):
        self.store = store
    
    async def save_snapshot(self, session_id: str, step_name: str, state: AgentState) -> None:
        snapshot_key = f"{session_id}:snapshot:{step_name}"
        await self.store.save(snapshot_key, state)
    
    async def load_snapshot(self, session_id: str, step_name: str) -> AgentState | None:
        snapshot_key = f"{session_id}:snapshot:{step_name}"
        return await self.store.load(snapshot_key)
    
    async def delete_snapshot(self, session_id: str, step_name: str) -> None:
        snapshot_key = f"{session_id}:snapshot:{step_name}"
        await self.store.delete(snapshot_key)
    
    async def list_snapshots(self, session_id: str) -> list[str]:
        # For in-memory, we'd need to iterate
        # This is a limitation - proper implementation needs store.list()
        # For now, return empty - override in specific stores
        return []