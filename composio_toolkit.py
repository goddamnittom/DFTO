from composio import ComposioToolSet
from composio.tools import tools
import logging

class ComposioIntegration:
    def __init__(self):
        self.toolset = ComposioToolSet()
        self.logger = logging.getLogger(__name__)
        self.available_tools = {}
        self._setup_tools()
    
    def _setup_tools(self):
        """Setup available tools."""
        try:
            # Register basic tools
            registered_tools = self.toolset.register_tools(
                tools=[
                    tools.CALCULATOR,
                    tools.FILETOOL,
                    # Add more tools as needed
                ]
            )
            
            for tool in registered_tools:
                self.available_tools[tool.name] = tool
                self.logger.info(f"Registered tool: {tool.name}")
                
        except Exception as e:
            self.logger.error(f"Error setting up Composio tools: {e}")
    
    def get_available_tools(self):
        """Get list of available tools."""
        return list(self.available_tools.keys())
    
    def execute_tool(self, tool_name: str, **kwargs):
        """Execute a specific tool."""
        if tool_name not in self.available_tools:
            raise ValueError(f"Tool {tool_name} not available")
        
        tool = self.available_tools[tool_name]
        # Execute tool with provided arguments
        return tool(**kwargs)