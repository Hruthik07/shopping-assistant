"""MCP (Model Context Protocol) client implementation."""

from typing import Dict, Any, List, Optional
from src.analytics.logger import logger


class MCPTool:
    """Base class for MCP tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool."""
        raise NotImplementedError

    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_parameters(),
        }

    def get_parameters(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {}


class MCPToolRegistry:
    """Registry for MCP tools."""

    def __init__(self):
        self.tools: Dict[str, MCPTool] = {}

    def register(self, tool: MCPTool):
        """Register a tool."""
        self.tools[tool.name] = tool
        logger.info(f"Registered MCP tool: {tool.name}")

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools."""
        return [tool.get_schema() for tool in self.tools.values()]


# Global tool registry
tool_registry = MCPToolRegistry()
