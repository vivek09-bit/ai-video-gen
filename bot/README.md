# Tube2D Telegram Bot & Razorpay Integration

A separate Telegram Bot integration for [Tube2DScript](..).

## Features
- **User Registration & Balance**: Tracks Telegram users and their credits in SQLite (`bot/bot.db`).
- **Razorpay Payment Gateway**: Users can purchase credit packs using Razorpay Payment Links (supports UPI, Cards, NetBanking).
- **Automated Webhooks**: Automatically validates Razorpay signature and credits user accounts on successful transaction.
- **Background Video Generation Queue**: Queues video requests, triggers Remotion rendering via `scripts/new_video.py`, and sends the completed MP4 directly into the user's Telegram chat.

## Required Packages
Install python dependencies in your environment:
```powershell
.venv\Scripts\pip install aiogram razorpay fastapi uvicorn aiosqlite
```

## Configuration (`.env`)
Add the following to your root `.env` or create `bot/.env`:
```env
# Telegram Bot Token (from @BotFather)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token

# Razorpay Credentials (from Razorpay Dashboard)
RAZORPAY_KEY_ID=rzp_live_or_test_key_id
RAZORPAY_KEY_SECRET=rzp_live_or_test_key_secret
RAZORPAY_WEBHOOK_SECRET=your_webhook_secret

# Webhook Server Settings
WEBHOOK_HOST=0.0.0.0
WEBHOOK_PORT=8000
```

## Running the Bot
```powershell
.venv\Scripts\python -m bot.main
```
