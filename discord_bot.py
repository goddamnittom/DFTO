import discord
from discord.ext import commands
import asyncio
import logging
from agent_core import ZeroClawAgent

class ZeroClawDiscordBot:
    def __init__(self, agent: ZeroClawAgent):
        self.agent = agent
        self.config = agent.config
        self.bot = commands.Bot(
            command_prefix='/',
            intents=discord.Intents.all(),
            description="ZeroClaw Autonomous AI Agent"
        )
        self.setup_commands()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def setup_commands(self):
        """Setup custom Discord commands."""
        
        @self.bot.event
        async def on_ready():
            print(f'{self.bot.user} has connected to Discord!')
            print(f'Guilds: {[guild.name for guild in self.bot.guilds]}')
        
        @self.bot.command(name='think', help='Ask ZeroClaw to think about something')
        async def think(ctx, *, query: str):
            if ctx.author.id != self.config["discord"]["owner_id"]:
                await ctx.send("❌ Only the owner can use this command.")
                return
            
            await ctx.send("🧠 Thinking...")
            response = await self.agent.execute_task(query)
            await ctx.send(f"🤖 **ZeroClaw Response:**\n{response}")
        
        @self.bot.command(name='status', help='Get ZeroClaw status')
        async def status(ctx):
            if ctx.author.id != self.config["discord"]["owner_id"]:
                await ctx.send("❌ Only the owner can use this command.")
                return
            
            status_info = (
                f"✅ **ZeroClaw Status**\n"
                f"• Model: {self.agent.config['model']['name']}\n"
                f"• Memory Length: {len(self.agent.conversation_history)}\n"
                f"• Tools Available: Calculator, FileTool\n"
                f"• Personality: {self.agent.config['agent']['personality'][:50]}..."
            )
            await ctx.send(status_info)
        
        @self.bot.command(name='memory', help='View recent conversation history')
        async def memory(ctx, count: int = 5):
            if ctx.author.id != self.config["discord"]["owner_id"]:
                await ctx.send("❌ Only the owner can use this command.")
                return
            
            if not self.agent.conversation_history:
                await ctx.send("📝 No conversation history available.")
                return
            
            recent_memory = self.agent.conversation_history[-count:]
            memory_text = "📝 **Recent Conversation History:**\n"
            
            for i, entry in enumerate(recent_memory, 1):
                role = entry['role'].upper()
                content = entry['content'][:100] + "..." if len(entry['content']) > 100 else entry['content']
                timestamp = entry['timestamp'].split('T')[1][:8]  # Time only
                memory_text += f"\n{i}. [{role}] {timestamp}: {content}\n"
            
            await ctx.send(memory_text)
        
        @self.bot.command(name='execute', help='Execute a specific task')
        async def execute(ctx, *, task: str):
            if ctx.author.id != self.config["discord"]["owner_id"]:
                await ctx.send("❌ Only the owner can use this command.")
                return
            
            await ctx.send("⚙️ Executing task...")
            result = await self.agent.execute_task(task)
            await ctx.send(f"🔧 **Execution Result:**\n{result}")
        
        @self.bot.command(name='tools', help='List available tools')
        async def tools_list(ctx):
            if ctx.author.id != self.config["discord"]["owner_id"]:
                await ctx.send("❌ Only the owner can use this command.")
                return
            
            tools_info = (
                "🛠️ **Available Tools:**\n"
                "• `calculator` - Perform mathematical calculations\n"
                "• `filetool` - Read files (with permission)\n"
                "• More tools can be added via Composio integration"
            )
            await ctx.send(tools_info)
        
        @self.bot.command(name='reset', help='Reset conversation memory')
        async def reset_memory(ctx):
            if ctx.author.id != self.config["discord"]["owner_id"]:
                await ctx.send("❌ Only the owner can use this command.")
                return
            
            self.agent.conversation_history.clear()
            await ctx.send("🔄 Conversation memory has been reset.")
        
        @self.bot.event
        async def on_message(message):
            # Don't respond to own messages
            if message.author == self.bot.user:
                return
            
            # Only process messages from owner in DM or specific channels
            if (message.author.id == self.config["discord"]["owner_id"] and 
                (isinstance(message.channel, discord.DMChannel) or 
                 message.channel.name.startswith('zeroclawn'))):
                
                await message')
                
                # Process the message through the agent
                response = await self.agent.execute_task(message.content)
                
                # Split long responses into chunks
                chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]
                for chunk in chunks:
                    await message.reply(chunk)
            
            # Process bot commands
            await self.bot.process_commands(message)
    
    async def start(self):
        """Start the Discord bot."""
        await self.bot.start(self.config["discord"]["token"])