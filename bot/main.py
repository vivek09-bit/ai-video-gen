import asyncio
import logging
import sys
import uvicorn
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage

from bot.config import BOT_TOKEN, WEBHOOK_HOST, WEBHOOK_PORT
from bot.db import init_db
from bot.handlers import router
from bot.worker import worker_loop
from bot.webhook import app as fastapi_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tube2d_bot")

async def run_fastapi():
    config = uvicorn.Config(fastapi_app, host=WEBHOOK_HOST, port=WEBHOOK_PORT, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()

async def main():
    if not BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN is missing in environment variables.")
        sys.exit(1)

    # Initialize SQLite tables
    await init_db()
    logger.info("Database initialized successfully.")

    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)

    # Run background tasks: worker consumer & webhook server
    asyncio.create_task(worker_loop())
    asyncio.create_task(run_fastapi())

    logger.info(f"Webhook server running on {WEBHOOK_HOST}:{WEBHOOK_PORT}")
    logger.info("Starting Telegram Bot Polling...")

    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())
