import os
from pathlib import Path
from dotenv import load_dotenv

# Load root .env first, then bot/.env if present
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")

# Telegram Bot
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# Razorpay
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")
RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")

# Webhook Server
WEBHOOK_HOST = os.getenv("WEBHOOK_HOST", "0.0.0.0")
WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "8000"))
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")  # e.g., https://your-domain.ngrok.app

# Database & Storage
DB_PATH = os.getenv("BOT_DB_PATH", str(Path(__file__).resolve().parent / "bot.db"))
OUT_DIR = ROOT_DIR / "out"

# Pricing & Credit Tiers (amount in INR, credits given)
PRICING_PLANS = {
    "plan_starter": {
        "title": "Starter Pack",
        "price_inr": 49,
        "credits": 5,
        "description": "5 Video Credits"
    },
    "plan_pro": {
        "title": "Pro Creator",
        "price_inr": 99,
        "credits": 12,
        "description": "12 Video Credits (Best Value)"
    },
    "plan_business": {
        "title": "Agency Pack",
        "price_inr": 199,
        "credits": 30,
        "description": "30 Video Credits"
    }
}

# Free trial credits on first registration
SIGNUP_BONUS_CREDITS = int(os.getenv("SIGNUP_BONUS_CREDITS", "1"))
# Credit cost per video minute
CREDIT_COST_PER_MINUTE = 1
