import json
from fastapi import FastAPI, Request, Header, HTTPException
from aiogram import Bot
from bot.config import RAZORPAY_WEBHOOK_SECRET, BOT_TOKEN
from bot.payments import verify_webhook_signature
from bot.db import mark_payment_completed

app = FastAPI(title="Razorpay Webhook Listener")

@app.post("/webhook/razorpay")
async def razorpay_webhook(
    request: Request,
    x_razorpay_signature: str = Header(None)
):
    body_bytes = await request.body()
    
    if RAZORPAY_WEBHOOK_SECRET:
        if not x_razorpay_signature or not verify_webhook_signature(body_bytes, x_razorpay_signature, RAZORPAY_WEBHOOK_SECRET):
            raise HTTPException(status_code=400, detail="Invalid signature")

    payload = json.loads(body_bytes.decode("utf-8"))
    event = payload.get("event")

    # Payment link or order paid event
    if event in ("payment_link.paid", "order.paid", "payment.captured"):
        entity = payload.get("payload", {}).get("payment_link", {}).get("entity") or \
                 payload.get("payload", {}).get("payment", {}).get("entity")
        
        payment_id = entity.get("id") if entity else None
        
        if payment_id:
            payment_info = await mark_payment_completed(payment_id)
            if payment_info:
                # Notify User on Telegram
                telegram_id = payment_info["telegram_id"]
                credits_added = payment_info["credits_added"]
                amount = payment_info["amount_inr"]
                
                bot = Bot(token=BOT_TOKEN)
                try:
                    await bot.send_message(
                        chat_id=telegram_id,
                        text=f"🎉 *Payment Successful!*\n\nAmount: ₹{amount}\nCredits Added: *+{credits_added}*\n\nYou can now use /create to make your videos!",
                        parse_mode="Markdown"
                    )
                finally:
                    await bot.session.close()

    return {"status": "ok"}
