import asyncio
import os
import sys
import uuid
from pathlib import Path
from aiogram import Bot, types
from bot.config import ROOT_DIR, OUT_DIR, BOT_TOKEN
from bot.db import update_job_status, add_user_credits

# Priority queue for video generation jobs
job_queue = asyncio.Queue()

async def worker_loop():
    bot = Bot(token=BOT_TOKEN)
    while True:
        job = await job_queue.get()
        job_id = job["job_id"]
        chat_id = job["telegram_id"]
        topic = job["topic"]
        duration = job["duration_minutes"]
        genre = job["genre"]
        credits_used = job["credits_used"]

        status_msg = await bot.send_message(
            chat_id=chat_id,
            text=f"🎬 *Video Generation Started!*\n\n*Topic:* {topic}\n*Duration:* {duration} min\n*Genre:* {genre}\n\n_Writing script, generating visuals, narration & rendering... This may take a few minutes._",
            parse_mode="Markdown"
        )

        out_video_name = f"video_{job_id}.mp4"
        out_path = OUT_DIR / out_video_name
        python_exe = sys.executable

        # Execute scripts/new_video.py
        cmd = [
            python_exe,
            str(ROOT_DIR / "scripts" / "new_video.py"),
            "--topic", topic,
            "--minutes", str(duration),
            "--genre", genre,
            "--free",
            "--out", str(out_path)
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(ROOT_DIR)
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0 and out_path.exists():
                await update_job_status(job_id, status="completed", output_file=str(out_path))
                
                # Send the rendered video back to user
                video_file = types.FSInputFile(str(out_path))
                await bot.send_video(
                    chat_id=chat_id,
                    video=video_file,
                    caption=f"✨ *Your AI Video is Ready!*\n\n*Topic:* {topic}\n*Duration:* {duration} min\n*Genre:* {genre}",
                    parse_mode="Markdown"
                )
            else:
                err_text = stderr.decode()[-300:] if stderr else "Unknown error during rendering."
                await update_job_status(job_id, status="failed", error_message=err_text)
                
                # Refund credits on failure
                await add_user_credits(chat_id, credits_used)
                await bot.send_message(
                    chat_id=chat_id,
                    text=f"⚠️ *Video generation failed.*\n\n{err_text}\n\nYour *{credits_used} credits* have been refunded.",
                    parse_mode="Markdown"
                )
        except Exception as e:
            await update_job_status(job_id, status="failed", error_message=str(e))
            await add_user_credits(chat_id, credits_used)
            await bot.send_message(
                chat_id=chat_id,
                text=f"⚠️ *An error occurred:*\n`{str(e)}`\n\nYour credits have been refunded.",
                parse_mode="Markdown"
            )
        finally:
            job_queue.task_done()
            await bot.session.close()

async def submit_video_job(telegram_id: int, topic: str, duration_minutes: int, genre: str, credits_used: int) -> str:
    job_id = uuid.uuid4().hex[:8]
    job_data = {
        "job_id": job_id,
        "telegram_id": telegram_id,
        "topic": topic,
        "duration_minutes": duration_minutes,
        "genre": genre,
        "credits_used": credits_used
    }
    await job_queue.put(job_data)
    return job_id
