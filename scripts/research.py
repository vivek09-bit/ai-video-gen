#!/usr/bin/env python3
"""Automated Topic & Trend Researcher for Viral YouTube, Reels & TikTok Videos.

Features:
  1. Live Search Autocomplete (YouTube, Google, Google News) - Real-time keyword & autocomplete intelligence.
  2. Viral Hook & Angle Ideation - High-CTR title hooks, psychological curiosity triggers, thumbnail concepts.
  3. AI-Powered Trend & Audience Expansion - Synthesizes high-retention scripts ideas for specific niches.
  4. Instant Pipeline Integration - Feed researched topic & keywords directly into scripts/new_video.py.

Usage:
  # Discover trending topics and keywords for a niche/keyword
  .venv\\Scripts\\python.exe scripts\\research.py "horror stories" --genre horror

  # Research in Hindi / Hinglish for Shorts & Reels
  .venv\\Scripts\\python.exe scripts\\research.py "scam alert" --genre scifi --lang hindi

  # Pick top trending topic and generate video immediately
  .venv\\Scripts\\python.exe scripts\\research.py "creepy facts" --genre horror --auto-generate --format reel --seconds 30
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_images import load_env_file  # noqa: E402
import beats  # noqa: E402
import metadata  # noqa: E402

load_env_file(ROOT / ".env")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def call_llm(prompt: str, model: str | None = None) -> str | None:
    """Call configured LLM (OpenRouter/Omniroute, DeepSeek, Fireworks, etc.) to get AI completion."""
    from new_video import resolve_llm_writer

    writer_info = resolve_llm_writer(requested_model=model)
    if not writer_info:
        return None

    _label, base_url, api_key, model_to_use = writer_info

    payload = json.dumps(
        {
            "model": model_to_use,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an elite viral YouTube & TikTok video growth strategist. Respond only with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
        }
    ).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) tube2dscript/1.0",
        "HTTP-Referer": "https://github.com/tube2dscript",
        "X-Title": "tube2dscript-research",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        f"{base_url}/chat/completions", data=payload, headers=headers, method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
    except Exception:
        return None

def fetch_youtube_suggestions(query: str, lang: str = "en", country: str = "IN") -> list[str]:
    """Fetch live real-time autocomplete suggestions from YouTube search engine."""
    clean_q = query.strip()
    if not clean_q:
        return []

    url = f"https://suggestqueries.google.com/complete/search?client=firefox&ds=yt&hl={lang}&gl={country}&q={urllib.parse.quote(clean_q)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            if isinstance(data, list) and len(data) > 1 and isinstance(data[1], list):
                return [str(s).strip() for s in data[1] if str(s).strip()]
    except Exception:
        pass
    return []


def fetch_google_suggestions(query: str, lang: str = "en", country: str = "IN") -> list[str]:
    """Fetch live search queries from Google Search Autocomplete."""
    clean_q = query.strip()
    if not clean_q:
        return []

    url = f"https://suggestqueries.google.com/complete/search?client=firefox&hl={lang}&gl={country}&q={urllib.parse.quote(clean_q)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            if isinstance(data, list) and len(data) > 1 and isinstance(data[1], list):
                return [str(s).strip() for s in data[1] if str(s).strip()]
    except Exception:
        pass
    return []


def generate_keyword_clusters(seed_query: str, genre: str | None = None, lang: str = "hi") -> dict:
    """Cluster seed keyword with modifiers: questions, vs/comparisons, long-tail, and viral prefixes."""
    seed = seed_query.strip()
    
    # 1. Base query suggestions
    yt_direct = fetch_youtube_suggestions(seed)
    google_direct = fetch_google_suggestions(seed)

    # 2. Modifiers for high-intent questions & curiosities
    modifiers = [
        f"what happens if {seed}",
        f"why do {seed}",
        f"real story {seed}",
        f"mystery of {seed}",
        f"top 5 {seed}",
        f"{seed} in hindi",
        f"{seed} facts",
        f"{seed} explained",
    ] if lang != "hi" else [
        f"{seed} kya hai",
        f"{seed} real story",
        f"{seed} hindi kahani",
        f"{seed} sach ya jhooth",
        f"{seed} rahasya",
        f"{seed} facts in hindi",
        f"{seed} viral video",
        f"{seed} mysterious incident",
    ]

    expanded_keywords: set[str] = set(yt_direct + google_direct)
    question_keywords: set[str] = set()

    for mod in modifiers[:4]:
        for res in fetch_youtube_suggestions(mod):
            if any(q in res.lower() for q in ("kya", "kyun", "kaise", "what", "why", "how", "mystery", "rahasya", "story")):
                question_keywords.add(res)
            else:
                expanded_keywords.add(res)

    # Clean duplicates & prioritize
    clean_expanded = [k for k in expanded_keywords if k.lower() != seed.lower()][:15]
    clean_questions = list(question_keywords)[:10]

    return {
        "seed": seed,
        "youtube_trending": yt_direct[:10],
        "google_search": google_direct[:8],
        "questions_and_curiosity": clean_questions,
        "long_tail_keywords": clean_expanded[:12],
    }


# --------------------------------------------------------------------------
# AI-Powered High-CTR Viral Angle & Topic Synthesizer
# --------------------------------------------------------------------------

VIRAL_PROMPT_TEMPLATE = """You are a master YouTube & TikTok algorithmic growth strategist who crafts viral, high-retention video ideas.

Given the Seed Topic: "{seed}"
Genre: "{genre}"
Target Format: "{format}" (Shorts/Reel or Long-form)
Language: "{language}"

Live Real-Time Autocomplete Keywords from YouTube & Google:
{keyword_context}

Generate 5 distinct high-CTR, viral video concepts. Each concept must maximize watch time, curiosity gaps, and retention.

Return ONLY a valid JSON object formatted exactly as:
{{
  "concepts": [
    {{
      "title": "Unstoppable High-CTR Clickable Title (under 60 chars)",
      "hook": "Opening 0-3s attention-grabbing audio line that stops scrolling",
      "core_angle": "The psychological twist / story tension in 1 sentence",
      "recommended_topic_arg": "Concise topic prompt ready for scripts/new_video.py",
      "target_emotion": "shock | curiosity | dread | excitement | wonder",
      "estimated_retention_score": "95%"
    }}
  ],
  "seo_tags": ["top 10 trending search tags for metadata"],
  "recommended_hashtags": ["#Tag1", "#Tag2", "#Tag3", "#Tag4", "#Tag5"]
}}
"""


def research_viral_angles(
    seed: str,
    genre: str = "horror",
    format_type: str = "reel",
    lang: str = "hindi",
    model: str = "omniroute",
) -> dict:
    """Run real-time keyword discovery and synthesize high-CTR video angles."""
    # 1. Fetch live keywords
    clusters = generate_keyword_clusters(seed, genre=genre, lang="hi" if "hi" in lang.lower() else "en")
    
    keyword_summary = (
        f"- Top YouTube Searches: {', '.join(clusters['youtube_trending'][:6])}\n"
        f"- Top Google Searches: {', '.join(clusters['google_search'][:5])}\n"
        f"- Audience Questions: {', '.join(clusters['questions_and_curiosity'][:5])}\n"
        f"- Long-Tail Searches: {', '.join(clusters['long_tail_keywords'][:6])}"
    )

    prompt = VIRAL_PROMPT_TEMPLATE.format(
        seed=seed,
        genre=genre,
        format=format_type,
        language=lang,
        keyword_context=keyword_summary,
    )

    # 2. Call AI via LLM helper
    ai_response = call_llm(prompt, model=model)
    parsed = {}
    if ai_response:
        try:
            # Extract JSON block
            match = re.search(r"\{.*\}", ai_response, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
        except Exception:
            pass

    # Fallback if LLM did not return structured concepts
    if not parsed or "concepts" not in parsed:
        parsed = {
            "concepts": [
                {
                    "title": f"The Dark Truth of {seed.title()}",
                    "hook": f"You won't believe what they found behind {seed}...",
                    "core_angle": f"A chilling mystery surrounding {seed} that nobody talks about.",
                    "recommended_topic_arg": f"{seed} mystery and dark secret",
                    "target_emotion": "curiosity",
                    "estimated_retention_score": "90%",
                },
                {
                    "title": f"Never Search About {seed.title()} At 3 AM",
                    "hook": f"If you think {seed} is normal, listen to this.",
                    "core_angle": f"Shocking revelations and eerie incidents connected to {seed}.",
                    "recommended_topic_arg": f"unexplained incidents of {seed}",
                    "target_emotion": "shock",
                    "estimated_retention_score": "92%",
                }
            ],
            "seo_tags": clusters["youtube_trending"][:8] or [seed],
            "recommended_hashtags": [f"#{seed.replace(' ', '')}", "#viral", "#trending", "#mystery"],
        }

    return {
        "seed": seed,
        "genre": genre,
        "keywords": clusters,
        "ai_synthesis": parsed,
    }


def format_research_report(report: dict) -> str:
    """Format research findings into a clean CLI dashboard."""
    seed = report["seed"]
    genre = report["genre"]
    kw = report["keywords"]
    ai = report["ai_synthesis"]

    lines = [
        "=" * 76,
        f"TREND & KEYWORD RESEARCH REPORT: '{seed.upper()}' ({genre.upper()})",
        "=" * 76,
        "\n1. REAL-TIME SEARCH AUTOCOMPLETE (Live YouTube & Google Data):",
        "  [YouTube Trending Queries]",
    ]
    for q in kw.get("youtube_trending", [])[:8]:
        lines.append(f"    - {q}")

    if kw.get("questions_and_curiosity"):
        lines.append("\n  [Audience Questions & Curiosity Searches]")
        for q in kw.get("questions_and_curiosity", [])[:6]:
            lines.append(f"    - {q}")

    if kw.get("long_tail_keywords"):
        lines.append("\n  [High-Intent Long-Tail Keywords]")
        for q in kw.get("long_tail_keywords", [])[:6]:
            lines.append(f"    - {q}")

    lines.extend([
        "\n" + "=" * 76,
        "2. VIRAL HIGH-RETENTION VIDEO CONCEPTS & ANGLES (AI Synthesized):",
        "=" * 76,
    ])

    for i, concept in enumerate(ai.get("concepts", []), 1):
        lines.extend([
            f"\n  #{i} [{concept.get('target_emotion', 'viral').upper()}] {concept.get('title', '')}",
            f"     Hook (0-3s) : \"{concept.get('hook', '')}\"",
            f"     Story Angle : {concept.get('core_angle', '')}",
            f"     CLI Topic   : --topic \"{concept.get('recommended_topic_arg', '')}\"",
            f"     Retention   : {concept.get('estimated_retention_score', '90+')}",
        ])

    if ai.get("seo_tags"):
        lines.extend([
            "\n3. RECOMMENDED SEO SEARCH TAGS:",
            f"   {', '.join(ai.get('seo_tags', [])[:10])}",
        ])

    if ai.get("recommended_hashtags"):
        lines.extend([
            "\n4. VIRAL HASHTAG SET:",
            f"   {' '.join(ai.get('recommended_hashtags', [])[:8])}",
        ])

    lines.append("\n" + "=" * 76)
    return "\n".join(lines)


DEFAULT_GENRE_SEEDS: dict[str, str] = {
    "horror": "hindi horror stories real incident",
    "scifi": "future science technology ai mystery",
    "documentary": "unsolved ancient mystery case study in hindi",
    "mythology": "ancient indian mythology secret rahasya",
    "explainer": "how it works science facts in hindi",
    "drama": "heart touching true incident story in hindi",
    "comedy": "funny real life moments comedy story",
    "motivational": "inspirational real life success story",
    "anime": "anime dark powers mystery",
    "fantasy": "dark fantasy kingdom mystery",
}


# --------------------------------------------------------------------------
# CLI Entrypoint
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automated Topic, Trending Keywords & Viral Angle Researcher with 1-Click Video & SEO Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Seed topic, keyword, or niche (e.g. 'horror story', 'ancient mystery'). If omitted, automatically uses top trending seeds for the genre.",
    )
    parser.add_argument("--genre", default="horror", choices=list(beats.GENRES.keys()), help="Video genre niche")
    parser.add_argument("--format", default="landscape", choices=["reel", "portrait", "landscape", "square"], help="Target video format")
    parser.add_argument("--lang", default="hindi", help="Language for search suggestions and viral hooks (hindi, hinglish, en)")
    parser.add_argument("--model", default="omniroute", help="LLM for viral angle synthesis (omniroute, deepseek, ollama)")
    parser.add_argument("--minutes", type=float, default=None, help="Target duration in minutes (e.g. --minutes 5, --minutes 1)")
    parser.add_argument("--seconds", type=float, default=None, help="Target duration in seconds (overrides --minutes)")
    parser.add_argument("--voice", default="spectral_child", help="Narration voice preset or blend (e.g. spectral_child, ghost_whisper, hm_omega)")
    parser.add_argument("--caption-bottom", type=float, default=0.40, help="Distance from bottom of screen for subtitles (e.g. 0.40 = 40%%)")
    parser.add_argument("--upload", "--publish", dest="upload", action="store_true", help="Automatically upload rendered video to YouTube with 7-tier SEO")
    parser.add_argument("--upload-platforms", default="youtube", help="Platforms to upload to (default: youtube)")
    parser.add_argument("--privacy", default="public", choices=["public", "unlisted", "private"], help="YouTube privacy status")
    parser.add_argument("--out", default=None, help="Output video file path")
    parser.add_argument(
        "--auto-generate",
        "--auto",
        dest="auto_generate",
        action="store_true",
        default=True,
        help="(default) Automatically research, write, voice, render, and SEO-tag the video",
    )
    parser.add_argument(
        "--research-only",
        dest="research_only",
        action="store_true",
        help="Only perform research and display findings without generating video",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON data")
    parser.add_argument("--no-title", dest="show_headline", action="store_false", default=False, help="Hide title overlays on video screen")
    parser.add_argument("--dynamic-speed", action="store_true", default=True, help="Dynamically modulate narration speed for dramatic tension")

    args = parser.parse_args()

    seed_query = args.query or DEFAULT_GENRE_SEEDS.get(args.genre, f"{args.genre} stories in hindi")
    print(f"\nGathering live search intelligence for: '{seed_query}' ({args.genre})...")
    report = research_viral_angles(
        seed=seed_query,
        genre=args.genre,
        format_type=args.format,
        lang=args.lang,
        model=args.model,
    )

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    print(format_research_report(report))

    if not args.research_only and args.auto_generate:
        concepts = report["ai_synthesis"].get("concepts") or []
        if not concepts:
            print("error: no concepts synthesized to generate video.", file=sys.stderr)
            return 1

        top_concept = concepts[0]
        selected_topic = top_concept.get("recommended_topic_arg") or top_concept.get("title") or seed_query
        print(f"\n[AUTO-PIPELINE] Launching video generation for top viral topic:")
        print(f"  Topic: {selected_topic}")
        print(f"  Hook: {top_concept.get('hook')}\n")

        duration_tag = f"{int(args.minutes)}m" if args.minutes else (f"{int(args.seconds)}s" if args.seconds else "1m")
        out_path = args.out or f"out/{args.genre}_{args.format}_{duration_tag}.mp4"

        cmd = [
            str(ROOT / ".venv" / "Scripts" / "python.exe"),
            str(ROOT / "scripts" / "new_video.py"),
            "--topic", selected_topic,
            "--genre", args.genre,
            "--format", args.format,
            "--voice", args.voice,
            "--model", args.model or "omniroute",
            "--out", out_path,
        ]

        if args.seconds is not None:
            cmd.extend(["--seconds", str(args.seconds)])
        elif args.minutes is not None:
            cmd.extend(["--minutes", str(args.minutes)])
        else:
            cmd.extend(["--minutes", "1.0"])

        if args.caption_bottom is not None:
            cmd.extend(["--caption-bottom", str(args.caption_bottom)])

        if not args.show_headline:
            cmd.append("--no-title")

        if args.dynamic_speed:
            cmd.append("--dynamic-speed")

        if "hi" in args.lang.lower() or "hinglish" in args.lang.lower():
            cmd.extend(["--subtitles", "hinglish"])
            cmd.extend(["--language", "hi"])

        if args.upload:
            cmd.append("--upload")
            cmd.extend(["--platforms", args.upload_platforms])
            cmd.extend(["--privacy", args.privacy])

        import subprocess
        subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
