import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from composio import ComposioToolSet
from composio.tools import tools
import ollama
import datetime

class ZeroClawAgent:
    def __init__(self, config_path: str = "config.json"):
        self.config = self.load_config(config_path)
        self.composio_toolset = ComposioToolSet()
        self.conversation_history = []
        self.max_memory = self.config["agent"]["max_memory"]
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize tools
        self.initialize_tools()
        
   , path: str) -> Dict:
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def initialize_tools(self):
        """Initialize available tools using Composio."""
        try:
            # Register common tools
            self.composio_toolset.register_tools(
                tools=[tools.CALCULATOR, tools.FILETOOL]
            )
            self.logger.info("Composio tools initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing Composio tools: {e}")
    
    async def get_llm_response(self, prompt: str) -> str:
        """Get response from local LLM."""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ollama.chat(
                    model=self.config["model"]["name"],
                    messages=[
                        {
                            "role": "system", 
                            "content": f"You are {self.config['agent']['name']}. {self.config['agent']['personality']}"
                        },
                        *self.conversation_history[-5:],  # Last 5 interactions
                        {"role": "user", "content": prompt}
                    ],
                    options={
                        "temperature": self.config["model"]["temperature"],
                        "num_predict": self.config["model"]["max_tokens"]
                    }
                )
            )
            return response['message']['content']
        except Exception as e:
            self.logger.error(f"LLM error: {e}")
            return "Error getting response from language model"
    
    def add_to_memory(self, role: str, content: str):
        """Add interaction to conversation memory."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Trim memory if too long
        if len(self.conversation_history) > self.max_memory:
            self.conversation_history = self.conversation_history[-self.max_memory:]
    
    async def execute_task(self, task_description: str) -> str:
        """Execute a task using reasoning and tools."""
        self.logger.info(f"Executing task: {task_description}")
        
        # Plan the task
        planning_prompt = f"""
        Analyze this task: "{task_description}"
        
        Think step by step:
        1. What needs to be done?
        2. What tools might be required?
        3. What is the expected outcome?
        
        Provide your analysis in JSON format with keys: analysis, steps, tools_needed, expected_outcome
        """
        
        plan_response = await self.get_llm_response(planning_prompt)
        self.add_to_memory("assistant", f"Task analysis: {plan_response}")
        
        # Execute based on plan
        execution_prompt = f"""
        Execute this task: "{task_description}"
        
        Previous analysis: {plan_response}
        
        Use available tools when needed. If you need to use a tool, respond in this format:
        <tool_call>
        {{
            "tool_name": "tool_name",
            "arguments": {{"arg1": "value1"}}
        }}
        </tool_call>
        
        Otherwise, provide the direct answer.
        """
        
        result = await self.get_llm_response(execution_prompt)
        
        # Check if tool call was requested
        if "<tool_call>" in result and "</tool_call>" in result:
            tool_match = result.split("<tool_call>")[1].split("</tool_call>")[0]
            try:
                tool_request = json.loads(tool_match.strip())
                tool_result = await self.execute_tool(tool_request["tool_name"], tool_request["arguments"])
                
                # Get final response incorporating tool result
                final_prompt = f"""
                Task: {task_description}
                Tool result: {tool_result}
                
                Provide the final answer incorporating the tool result.
                """
                result = await self.get_llm_response(final_prompt)
            except json.JSONDecodeError:
                self.logger.warning("Invalid tool call format")
        
        self.add_to_memory("assistant", f"Task result: {result}")
        return result
    
    async def execute_tool(self, tool_name: str, arguments: Dict) -> str:
        """Execute a Composio tool."""
        try:
            # This is a simplified version - in practice, you'd use the actual Composio integration
            if tool_name.lower() == "calculator":
                # Simple calculator example
                expression = arguments.get("expression", "")
                try:
                    result = eval(expression)  # Note: In production, use safer evaluation
                    return f"Calculator result: {result}"
                except:
                    return "Calculator error: Invalid expression"
            elif tool_name.lower() == "filetool":
                action = arguments.get("action", "")
                if action == "read":
                    filename = arguments.get("filename", "")
                    try:
                        with open(filename, 'r') as f:
                            content = f.read()
                        return f"File content: {content[:500]}..."  # Limit output
                    except FileNotFoundError:
                        return "File not found"
                else:
                    return "Unsupported file operation"
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            self.logger.error(f"Tool execution error: {e}")
            return f"Tool execution failed: {str(e)}"
    
    async def autonomous_operation(self):
        """Run autonomous operations."""
        while True:
            try:
                # Example autonomous task - you can customize this
                autonomous_task = "Check system status and report any issues"
                result = await self.execute_task(autonomous_task)
                self.logger.info(f"Autonomous result: {result}")
                
                # Wait before next autonomous cycle
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                self.logger.error(f"Autonomous operation error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry