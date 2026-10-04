import aiosqlite
from datetime import datetime
from bot.config import DB_PATH, SIGNUP_BONUS_CREDITS

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                telegram_id INTEGER PRIMARY KEY,
                username TEXT,
                full_name TEXT,
                credits INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS payments (
                payment_id TEXT PRIMARY KEY,
                order_id TEXT,
                telegram_id INTEGER,
                amount_inr INTEGER,
                credits_added INTEGER,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (telegram_id) REFERENCES users (telegram_id)
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                telegram_id INTEGER,
                topic TEXT,
                duration_minutes INTEGER,
                genre TEXT,
                status TEXT,
                output_file TEXT,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (telegram_id) REFERENCES users (telegram_id)
            )
        """)
        await db.commit()

async def get_or_create_user(telegram_id: int, username: str | None, full_name: str | None):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM users WHERE telegram_id = ?", (telegram_id,)) as cursor:
            user = await cursor.fetchone()
            if user:
                return dict(user), False
        
        # New user creation with bonus credits
        await db.execute(
            "INSERT INTO users (telegram_id, username, full_name, credits) VALUES (?, ?, ?, ?)",
            (telegram_id, username or "", full_name or "", SIGNUP_BONUS_CREDITS)
        )
        await db.commit()
        
        async with db.execute("SELECT * FROM users WHERE telegram_id = ?", (telegram_id,)) as cursor:
            user = await cursor.fetchone()
            return dict(user), True

async def get_user(telegram_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM users WHERE telegram_id = ?", (telegram_id,)) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

async def add_user_credits(telegram_id: int, credits_to_add: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET credits = credits + ?, updated_at = CURRENT_TIMESTAMP WHERE telegram_id = ?",
            (credits_to_add, telegram_id)
        )
        await db.commit()

async def deduct_user_credits(telegram_id: int, credits_to_deduct: int) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT credits FROM users WHERE telegram_id = ?", (telegram_id,)) as cursor:
            row = await cursor.fetchone()
            if not row or row["credits"] < credits_to_deduct:
                return False
        
        await db.execute(
            "UPDATE users SET credits = credits - ?, updated_at = CURRENT_TIMESTAMP WHERE telegram_id = ?",
            (credits_to_deduct, telegram_id)
        )
        await db.commit()
        return True

async def create_payment_record(payment_id: str, order_id: str, telegram_id: int, amount_inr: int, credits_added: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO payments (payment_id, order_id, telegram_id, amount_inr, credits_added, status) VALUES (?, ?, ?, ?, ?, 'created')",
            (payment_id, order_id, telegram_id, amount_inr, credits_added)
        )
        await db.commit()

async def mark_payment_completed(payment_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM payments WHERE payment_id = ?", (payment_id,)) as cursor:
            payment = await cursor.fetchone()
            if not payment or payment["status"] == "paid":
                return None
        
        await db.execute("UPDATE payments SET status = 'paid' WHERE payment_id = ?", (payment_id,))
        await db.execute(
            "UPDATE users SET credits = credits + ?, updated_at = CURRENT_TIMESTAMP WHERE telegram_id = ?",
            (payment["credits_added"], payment["telegram_id"])
        )
        await db.commit()
        return dict(payment)

async def create_job_record(job_id: str, telegram_id: int, topic: str, duration_minutes: int, genre: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO jobs (job_id, telegram_id, topic, duration_minutes, genre, status) VALUES (?, ?, ?, ?, ?, 'pending')",
            (job_id, telegram_id, topic, duration_minutes, genre)
        )
        await db.commit()

async def update_job_status(job_id: str, status: str, output_file: str | None = None, error_message: str | None = None):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE jobs SET status = ?, output_file = ?, error_message = ?, completed_at = CURRENT_TIMESTAMP WHERE job_id = ?",
            (status, output_file, error_message, job_id)
        )
        await db.commit()
