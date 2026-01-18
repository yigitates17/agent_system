# registries/agent_registry.py

from core.base_agent import BaseAgent


class AgentRegistry:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._agents = {}
        return cls._instance
    
    def register(self, agent: BaseAgent) -> None:
        self._agents[agent.name] = agent
    
    def get(self, name: str) -> BaseAgent | None:
        return self._agents.get(name)
    
    def get_all(self) -> list[BaseAgent]:
        return list(self._agents.values())
    
    def get_names(self) -> list[str]:
        return list(self._agents.keys())
    
    def has(self, name: str) -> bool:
        return name in self._agents
    
    def remove(self, name: str) -> None:
        self._agents.pop(name, None)
    
    def clear(self) -> None:
        self._agents = {}