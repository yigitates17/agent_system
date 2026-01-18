# registries/tool_registry.py

from core.base_tool import BaseTool


class ToolRegistry:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
        return cls._instance
    
    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
    
    def register_many(self, tools: list[BaseTool]) -> None:
        for tool in tools:
            self.register(tool)
    
    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)
    
    def get_all(self) -> list[BaseTool]:
        return list(self._tools.values())
    
    def get_names(self) -> list[str]:
        return list(self._tools.keys())
    
    def has(self, name: str) -> bool:
        return name in self._tools
    
    def remove(self, name: str) -> None:
        self._tools.pop(name, None)
    
    def clear(self) -> None:
        self._tools = {}