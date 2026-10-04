from aiogram import Router, F
from aiogram.filters import CommandStart, Command
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton
)
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

from bot.config import PRICING_PLANS, CREDIT_COST_PER_MINUTE
from bot.db import (
    get_or_create_user,
    get_user,
    deduct_user_credits,
    create_job_record
)
from bot.payments import create_payment_link
from bot.worker import submit_video_job

router = Router()

class VideoCreationForm(StatesGroup):
    waiting_for_topic = State()
    waiting_for_genre = State()
    waiting_for_duration = State()

@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    user, is_new = await get_or_create_user(
        telegram_id=message.from_user.id,
        username=message.from_user.username,
        full_name=message.from_user.full_name
    )
    
    welcome_msg = (
        f"👋 *Welcome, {message.from_user.first_name}!*\n\n"
        f"🎬 *Tube2D AI Video Generator Bot*\n"
        f"Generate long-form narrated 2D animated videos instantly.\n\n"
        f"💰 *Your Balance:* `{user['credits']} Credits`\n"
    )
    if is_new:
        welcome_msg += "\n🎁 *You received 1 Free Welcome Credit!*\n"

    welcome_msg += (
        "\n*Commands:*\n"
        "• /create - Create a new AI video\n"
        "• /buy - Buy more credits (Razorpay)\n"
        "• /balance - Check your current balance\n"
        "• /help - Usage instructions\n"
    )

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🎬 Create Video", callback_data="start_create")],
        [InlineKeyboardButton(text="💳 Buy Credits", callback_data="show_plans")],
        [InlineKeyboardButton(text="💰 My Balance", callback_data="check_balance")]
    ])

    await message.answer(welcome_msg, parse_mode="Markdown", reply_markup=keyboard)

@router.message(Command("balance"))
@router.callback_query(F.data == "check_balance")
async def cmd_balance(event: Message | CallbackQuery):
    user_id = event.from_user.id
    user = await get_user(user_id)
    credits = user["credits"] if user else 0
    text = (
        f"💰 *Account Balance*\n\n"
        f"User: `{event.from_user.full_name}`\n"
        f"Credits: *{credits}*\n\n"
        f"ℹ️ _1 credit = 1 minute of generated video._"
    )
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="💳 Top Up Credits", callback_data="show_plans")],
        [InlineKeyboardButton(text="🎬 Generate Video", callback_data="start_create")]
    ])
    if isinstance(event, CallbackQuery):
        await event.message.edit_text(text, parse_mode="Markdown", reply_markup=keyboard)
        await event.answer()
    else:
        await event.answer(text, parse_mode="Markdown", reply_markup=keyboard)

@router.message(Command("buy"))
@router.callback_query(F.data == "show_plans")
async def cmd_buy(event: Message | CallbackQuery):
    buttons = []
    for plan_key, plan in PRICING_PLANS.items():
        btn_text = f"{plan['title']} - ₹{plan['price_inr']} ({plan['credits']} Credits)"
        buttons.append([InlineKeyboardButton(text=btn_text, callback_data=f"buy_{plan_key}")])

    keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
    text = (
        "💳 *Select a Credit Plan*\n\n"
        "Pay securely via *Razorpay (UPI / Cards / NetBanking)*:\n"
    )
    for p in PRICING_PLANS.values():
        text += f"• *{p['title']}*: ₹{p['price_inr']} for {p['credits']} Credits\n"

    if isinstance(event, CallbackQuery):
        await event.message.edit_text(text, parse_mode="Markdown", reply_markup=keyboard)
        await event.answer()
    else:
        await event.answer(text, parse_mode="Markdown", reply_markup=keyboard)

@router.callback_query(F.data.startswith("buy_"))
async def handle_buy_plan(callback: CallbackQuery):
    plan_key = callback.data.replace("buy_", "")
    user_id = callback.from_user.id

    try:
        payment = await create_payment_link(telegram_id=user_id, plan_key=plan_key)
        pay_url = payment["url"]
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🚀 Pay Now with Razorpay", url=pay_url)],
            [InlineKeyboardButton(text="🔙 Back to Plans", callback_data="show_plans")]
        ])

        await callback.message.edit_text(
            f"🛒 *Order Created*\n\n"
            f"*Plan:* {payment['plan_title']}\n"
            f"*Amount:* ₹{payment['amount']}\n"
            f"*Credits:* +{payment['credits']}\n\n"
            f"Click the button below to complete payment. Your credits will be added immediately upon confirmation.",
            parse_mode="Markdown",
            reply_markup=keyboard
        )
    except Exception as e:
        await callback.answer(f"Error: {str(e)}", show_alert=True)

@router.message(Command("create"))
@router.callback_query(F.data == "start_create")
async def start_video_creation(event: Message | CallbackQuery, state: FSMContext):
    user_id = event.from_user.id
    user = await get_user(user_id)
    
    if not user or user["credits"] < 1:
        text = "⚠️ *Insufficient Credits!*\nYou need at least 1 credit to generate a video."
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="💳 Buy Credits", callback_data="show_plans")]
        ])
        if isinstance(event, CallbackQuery):
            await event.message.edit_text(text, parse_mode="Markdown", reply_markup=keyboard)
            await event.answer()
        else:
            await event.answer(text, parse_mode="Markdown", reply_markup=keyboard)
        return

    text = "📝 *Step 1/3: Enter Topic or Prompt*\n\nSend the topic or story idea for the video."
    if isinstance(event, CallbackQuery):
        await event.message.edit_text(text, parse_mode="Markdown")
        await event.answer()
    else:
        await event.answer(text, parse_mode="Markdown")
    
    await state.set_state(VideoCreationForm.waiting_for_topic)

@router.message(VideoCreationForm.waiting_for_topic)
async def process_topic(message: Message, state: FSMContext):
    await state.update_data(topic=message.text)
    
    genres = ["fantasy", "scifi", "horror", "documentary", "mythology", "comedy", "drama", "motivational"]
    buttons = [[InlineKeyboardButton(text=g.capitalize(), callback_data=f"genre_{g}")] for g in genres]
    keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
    
    await message.answer("🎭 *Step 2/3: Select Genre*", parse_mode="Markdown", reply_markup=keyboard)
    await state.set_state(VideoCreationForm.waiting_for_genre)

@router.callback_query(VideoCreationForm.waiting_for_genre, F.data.startswith("genre_"))
async def process_genre(callback: CallbackQuery, state: FSMContext):
    genre = callback.data.replace("genre_", "")
    await state.update_data(genre=genre)

    user = await get_user(callback.from_user.id)
    user_credits = user["credits"] if user else 0

    durations = [1, 2, 3, 5]
    buttons = []
    for d in durations:
        if user_credits >= d:
            buttons.append([InlineKeyboardButton(text=f"{d} Minute ({d} Credit)", callback_data=f"duration_{d}")])

    keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
    await callback.message.edit_text(
        f"⏱️ *Step 3/3: Select Duration*\n\nYour Credits: *{user_credits}*",
        parse_mode="Markdown",
        reply_markup=keyboard
    )
    await state.set_state(VideoCreationForm.waiting_for_duration)
    await callback.answer()

@router.callback_query(VideoCreationForm.waiting_for_duration, F.data.startswith("duration_"))
async def process_duration(callback: CallbackQuery, state: FSMContext):
    duration = int(callback.data.replace("duration_", ""))
    data = await state.get_data()
    topic = data.get("topic")
    genre = data.get("genre")
    user_id = callback.from_user.id

    cost = duration * CREDIT_COST_PER_MINUTE
    deducted = await deduct_user_credits(user_id, cost)
    
    if not deducted:
        await callback.message.edit_text("⚠️ *Failed to deduct credits. Please check your balance.*", parse_mode="Markdown")
        await state.clear()
        return

    # Create job record and dispatch to worker queue
    job_id = await submit_video_job(
        telegram_id=user_id,
        topic=topic,
        duration_minutes=duration,
        genre=genre,
        credits_used=cost
    )
    await create_job_record(job_id, user_id, topic, duration, genre)

    await callback.message.edit_text(
        f"⏳ *Job Queued (#`{job_id}`)*\n\n"
        f"*Topic:* {topic}\n"
        f"*Genre:* {genre.capitalize()}\n"
        f"*Duration:* {duration} min\n"
        f"*Credits Deducted:* {cost}\n\n"
        f"The bot will send the video directly here as soon as rendering completes!",
        parse_mode="Markdown"
    )
    await state.clear()
    await callback.answer()
