import hmac
import hashlib
import razorpay
from bot.config import RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET, PRICING_PLANS
from bot.db import create_payment_record

def get_razorpay_client():
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        raise ValueError("Razorpay API Keys are not configured in environment variables.")
    return razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

async def create_payment_link(telegram_id: int, plan_key: str) -> dict:
    if plan_key not in PRICING_PLANS:
        raise ValueError("Invalid pricing plan selected.")

    plan = PRICING_PLANS[plan_key]
    client = get_razorpay_client()

    amount_paise = plan["price_inr"] * 100
    
    # Create Razorpay Payment Link
    link_data = {
        "amount": amount_paise,
        "currency": "INR",
        "accept_partial": False,
        "description": f"{plan['title']} - {plan['credits']} Video Credits",
        "notes": {
            "telegram_id": str(telegram_id),
            "plan_key": plan_key,
            "credits": str(plan["credits"])
        },
        "notify": {
            "sms": False,
            "email": False
        },
        "reminder_enable": False
    }
    
    response = client.payment_link.create(link_data)
    payment_link_id = response.get("id")
    short_url = response.get("short_url")

    # Record in local DB
    await create_payment_record(
        payment_id=payment_link_id,
        order_id=response.get("order_id", ""),
        telegram_id=telegram_id,
        amount_inr=plan["price_inr"],
        credits_added=plan["credits"]
    )

    return {
        "payment_id": payment_link_id,
        "url": short_url,
        "amount": plan["price_inr"],
        "credits": plan["credits"],
        "plan_title": plan["title"]
    }

def verify_webhook_signature(body_bytes: bytes, signature: str, secret: str) -> bool:
    if not secret:
        return False
    expected_signature = hmac.new(
        secret.encode("utf-8"),
        body_bytes,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected_signature, signature)
