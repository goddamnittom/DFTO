#!/usr/bin/env python3
"""
ZeroClaw Autonomous AI Agent for Raspberry Pi 5
"""

import asyncio
import signal
import sys
import logging
from agent_core import ZeroClawAgent
from discord_bot import ZeroClawDiscordBot

class ZeroClawMain:
    def __init__(self):
        self.agent = ZeroClawAgent()
        self.discord_bot = ZeroClawDiscordBot(self.agent)
        self.running = False
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    async def startup(self):
        """Initialize and start all components."""
        self.logger.info("🚀 Starting ZeroClaw Autonomous Agent...")
        
        # Start autonomous operations in background
        autonomous_task = asyncio.create_task(self.agent.autonomous_operation())
        
        # Start Discord bot
        try:
            await self.discord_bot.start()
        except KeyboardInterrupt:
            self.logger.info("🛑 Shutting down...")
        finally:
            self.running = False
            autonomous_task.cancel()
            try:
                await autonomous_task
            except asyncio.CancelledError:
                pass
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        sys.exit(0)

def main():
    # Set up signal handlers
    app = ZeroClawMain()
    signal.signal(signal.SIGINT, app.signal_handler)
    signal.signal(signal.SIGTERM, app.signal_handler)
    
    # Run the application
    try:
        asyncio.run(app.startup())
    except KeyboardInterrupt:
        print("\n👋 ZeroClaw shutting down gracefully...")

if __name__ == "__main__":
    main()