import os
from datetime import datetime
import discord
import logging
import warnings
from dotenv import load_dotenv
from src.engine.agent import ask_sentinel

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("discord").setLevel(logging.ERROR)

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

class SentinelBot(discord.Client):
    async def on_ready(self):
        print(f'--- Drexel Sentinel Online ---')
        print(f'Logged in as: {self.user}')
        print(f'Syncing with Gemini 2.0... Ready.')

    async def on_message(self, message):
        if message.author == self.user:
            return

        time_bucket = datetime.now().timestamp() // 1800 
        thread_id = f"user_{message.author.id}_{int(time_bucket)}"

        if self.user.mentioned_in(message) or isinstance(message.channel, discord.DMChannel):
            async with message.channel.typing():
                user_query = message.content.replace(f'<@!{self.user.id}>', '').strip()
                
                response = ask_sentinel(user_query, thread_id=thread_id)
                
                await message.reply(response)

intents = discord.Intents.default()
intents.message_content = True 

client = SentinelBot(intents=intents)

if __name__ == "__main__":
    client.run(DISCORD_TOKEN)