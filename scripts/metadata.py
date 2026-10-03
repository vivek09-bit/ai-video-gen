#!/usr/bin/env python3
"""Multi-Platform Social Media Metadata Generator (YouTube, Instagram, Facebook, TikTok).

Generates complete, SEO-optimized, platform-tailored upload metadata:
  - YouTube (Long-form & Shorts): Click-worthy SEO Titles (<100 chars), Structured Descriptions,
    Dynamic Chapter Timestamps (00:00 format), Search Tags array, Category ID, #Shorts flags.
  - Instagram (Reels & Feed): Hook lines, Spaced Captions with Emojis, Engagement CTAs,
    25-30 curated high-ranking Hashtags, Full copy-paste caption.
  - Facebook (Reels & Watch): Engaging Headlines (<255 chars), Discussion-starter Descriptions,
    3-5 Strategic Hashtags, Video Tags.
  - TikTok: Punchy caption (<150 chars), High-engagement viral Hashtags (#fyp, #viral, topic).

Supports dual AI (LLM) generation and intelligent procedural fallback.
Outputs machine-readable JSON (for automated uploaders / APIs) and formatted Markdown (for easy copy-pasting).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import beats  # noqa: E402
import transliterate  # noqa: E402

# --------------------------------------------------------------------------
# Platform Configuration & Hashtag Banks
# --------------------------------------------------------------------------

YOUTUBE_CATEGORY_MAP: dict[str, str] = {
    "scifi": "28",        # Science & Technology
    "documentary": "27",  # Education
    "motivational": "27", # Education
    "comedy": "23",       # Comedy
    "anime": "1",         # Film & Animation
    "fantasy": "1",       # Film & Animation
    "mythology": "24",    # Entertainment
    "drama": "24",        # Entertainment
    "horror": "24",       # Entertainment
}

GENRE_HASHTAG_BANKS: dict[str, dict[str, list[str]]] = {
    "anime": {
        "youtube": ["#Anime", "#AnimeEdit", "#AnimeShorts", "#Animation", "#ActionAnime", "#Otaku", "#AnimeCommunity", "#HindiAnime"],
        "instagram": [
            "#anime", "#animereels", "#animeedits", "#animeedit", "#otaku", "#manga",
            "#animelover", "#animeworld", "#animeart", "#animefans", "#animememes",
            "#shonen", "#actionanime", "#animefight", "#reelsindia", "#trendingreels",
            "#viralreels", "#explorepage", "#instaanime", "#hindianime", "#animestatus",
            "#animepage", "#animefreak", "#otakugirl", "#otakulife"
        ],
        "facebook": ["#Anime", "#AnimeAction", "#AnimeLovers", "#Animation", "#Reels"],
        "tiktok": ["#anime", "#animeedit", "#animereels", "#otaku", "#animefight", "#fyp", "#viral", "#foryou"],
    },
    "scifi": {
        "youtube": ["#SciFi", "#AI", "#Technology", "#FutureTech", "#Cyberpunk", "#ScienceFiction", "#Futuristic", "#TechNews", "#Shorts"],
        "instagram": [
            "#scifi", "#cyberpunk", "#futuretech", "#technology", "#artificialintelligence",
            "#ai", "#cybersecurity", "#techtrends", "#futuristic", "#scifimovies",
            "#technews", "#quantumcomputing", "#robotics", "#innovation", "#digitalfuture",
            "#reelsindia", "#trendingreels", "#viralreels", "#explorepage", "#scifiart",
            "#techreels", "#scifidaily", "#gadgets", "#techcommunity", "#future"
        ],
        "facebook": ["#SciFi", "#Technology", "#FutureTech", "#ArtificialIntelligence", "#TechTrends"],
        "tiktok": ["#scifi", "#cyberpunk", "#futuretech", "#ai", "#technology", "#fyp", "#viral", "#foryou"],
    },
    "fantasy": {
        "youtube": ["#Fantasy", "#Magic", "#FairyTale", "#Storytelling", "#FantasyStory", "#MagicalWorld", "#Animation", "#Shorts"],
        "instagram": [
            "#fantasy", "#fairytale", "#magic", "#magical", "#mythical", "#fantasyart",
            "#storytelling", "#storytime", "#fantasyworld", "#enchanted", "#fairytales",
            "#magickingdom", "#reelsindia", "#trendingreels", "#viralreels", "#explorepage",
            "#hindistories", "#kahaniya", "#magicworld", "#bedtimestories", "#fantasydaily",
            "#creativewriting", "#folktale", "#wonderland", "#cinematic"
        ],
        "facebook": ["#Fantasy", "#Magic", "#FairyTale", "#Storytelling", "#Kahaniya"],
        "tiktok": ["#fantasy", "#magic", "#storytime", "#fairytale", "#viral", "#fyp", "#foryou"],
    },
    "documentary": {
        "youtube": ["#Documentary", "#Facts", "#History", "#Science", "#DidYouKnow", "#Educational", "#Knowledge", "#Shorts"],
        "instagram": [
            "#documentary", "#facts", "#history", "#science", "#didyouknow", "#knowledge",
            "#educational", "#discovery", "#mysteries", "#worldhistory", "#learnsomethingnew",
            "#amazingfacts", "#factsdaily", "#interestingfacts", "#reelsindia", "#trendingreels",
            "#viralreels", "#explorepage", "#generalknowledge", "#mindblowingfacts", "#historyfacts",
            "#sciencefacts", "#realstories", "#infotainment", "#curiosity"
        ],
        "facebook": ["#Documentary", "#Facts", "#History", "#Knowledge", "#DidYouKnow"],
        "tiktok": ["#facts", "#history", "#didyouknow", "#documentary", "#learnontiktok", "#fyp", "#viral", "#foryou"],
    },
    "mythology": {
        "youtube": ["#Mythology", "#AncientHistory", "#IndianMythology", "#SanatanDharma", "#HinduMythology", "#EpicStories", "#Shorts"],
        "instagram": [
            "#mythology", "#ancienthistory", "#indianmythology", "#sanatandharma", "#hindumythology",
            "#ancientindia", "#epicstories", "#legend", "#mythological", "#historyofindia",
            "#mahabharat", "#ramayan", "#spiritual", "#ancientwisdom", "#reelsindia",
            "#trendingreels", "#viralreels", "#explorepage", "#hinduism", "#culture",
            "#ancienttales", "#sanatan", "#indianculture", "#folklore", "#divine"
        ],
        "facebook": ["#Mythology", "#IndianMythology", "#AncientHistory", "#SanatanDharma", "#History"],
        "tiktok": ["#mythology", "#ancienthistory", "#history", "#storytime", "#viral", "#fyp", "#foryou"],
    },
    "motivational": {
        "youtube": ["#Motivation", "#Success", "#Mindset", "#Inspiration", "#NeverGiveUp", "#HardWork", "#Discipline", "#Shorts"],
        "instagram": [
            "#motivation", "#success", "#mindset", "#inspiration", "#nevergiveup", "#hardwork",
            "#goals", "#discipline", "#positivevibes", "#entrepreneur", "#growth",
            "#dailyquotes", "#successquotes", "#inspirational", "#motivationalreels",
            "#reelsindia", "#trendingreels", "#viralreels", "#explorepage", "#lifelessons",
            "#selfimprovement", "#focus", "#grind", "#hustle", "#positivity"
        ],
        "facebook": ["#Motivation", "#Success", "#Inspiration", "#NeverGiveUp", "#Mindset"],
        "tiktok": ["#motivation", "#mindset", "#success", "#inspiration", "#grind", "#fyp", "#viral", "#foryou"],
    },
    "drama": {
        "youtube": ["#Drama", "#Storytelling", "#Emotional", "#Cinematic", "#ShortFilm", "#HumanStories", "#Shorts"],
        "instagram": [
            "#drama", "#storytelling", "#emotional", "#cinematic", "#shortfilm", "#lifequotes",
            "#deepthoughts", "#feelings", "#relationships", "#humanstories", "#reallife",
            "#shortstory", "#heartfelt", "#reelsindia", "#trendingreels", "#viralreels",
            "#explorepage", "#storytime", "#hindistories", "#kahani", "#relatable",
            "#emotions", "#lifequotes", "#cinematicvideo", "#filmmaking"
        ],
        "facebook": ["#Drama", "#Storytelling", "#Emotional", "#LifeLessons", "#ShortFilm"],
        "tiktok": ["#drama", "#storytime", "#emotional", "#cinematic", "#viral", "#fyp", "#foryou"],
    },
    "comedy": {
        "youtube": ["#Comedy", "#Funny", "#Humor", "#Jokes", "#Meme", "#FunnyVideo", "#DesiComedy", "#Shorts"],
        "instagram": [
            "#comedy", "#funny", "#humor", "#jokes", "#meme", "#funnymemes", "#laugh",
            "#relatable", "#viralhumor", "#desimemes", "#dailycomedy", "#funnyvideos",
            "#chutkule", "#hastehaste", "#reelsindia", "#trendingreels", "#viralreels",
            "#explorepage", "#memesdaily", "#bakchodi", "#comedyreels", "#standupcomedy",
            "#funnymoments", "#dankmemes", "#hilarious"
        ],
        "facebook": ["#Comedy", "#Funny", "#Humor", "#Jokes", "#Memes"],
        "tiktok": ["#comedy", "#funny", "#jokes", "#humor", "#memes", "#fyp", "#viral", "#foryou"],
    },
    "horror": {
        "youtube": ["#Horror", "#Suspense", "#Mystery", "#Creepy", "#Haunted", "#GhostStories", "#Thriller", "#Shorts"],
        "instagram": [
            "#horror", "#suspense", "#mystery", "#creepy", "#haunted", "#ghoststories",
            "#thriller", "#scary", "#paranormal", "#darkstories", "#horrormovies",
            "#bhootiya", "#khauf", "#darr", "#reelsindia", "#trendingreels", "#viralreels",
            "#explorepage", "#horrorstories", "#scarystories", "#darkart", "#supernatural",
            "#nightmare", "#creepyfacts", "#horrorgram"
        ],
        "facebook": ["#Horror", "#Suspense", "#Mystery", "#Haunted", "#ScaryStories"],
        "tiktok": ["#horror", "#scary", "#creepy", "#ghoststories", "#mystery", "#fyp", "#viral", "#foryou"],
    },
}

# --------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------

def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS format (or HH:MM:SS for longer videos)."""
    total_sec = max(0, int(round(seconds)))
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    secs = total_sec % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def extract_chapters_from_data(
    script_data: dict, timings_data: dict | None = None
) -> list[tuple[str, str]]:
    """Extract chapter timestamps and titles [(00:00, Title), ...].

    Ensures 00:00 is always present for valid YouTube chapters.
    """
    scenes = script_data.get("scenes", [])
    if not scenes:
        return [("00:00", "Introduction")]

    timing_scenes = timings_data.get("scenes", []) if timings_data else []
    chapters = []

    for i, scene in enumerate(scenes):
        headline = scene.get("headline", f"Scene {i + 1}").strip()
        # Clean headline from visual tags
        if " | VISUAL:" in headline or " | visual:" in headline.lower():
            headline = re.split(r"\s*\|\s*visual:\s*", headline, flags=re.IGNORECASE)[0].strip()

        # If headline is in Devanagari, also create a clean readable version
        clean_title = headline if headline else f"Scene {i + 1}"

        # Get timing
        if i == 0:
            ts_str = "00:00"
        elif i < len(timing_scenes):
            ts_sec = timing_scenes[i].get("start", 0.0)
            ts_str = format_timestamp(ts_sec)
        else:
            # Estimate roughly based on scene index (e.g. 15s per scene)
            ts_str = format_timestamp(i * 15.0)

        chapters.append((ts_str, clean_title))

    # Ensure the first chapter is always at 00:00
    if chapters and chapters[0][0] != "00:00":
        chapters[0] = ("00:00", chapters[0][1])

    return chapters


def build_seo_tags(topic: str, genre: str, language: str, headlines: list[str]) -> list[str]:
    """Generate search keywords and tags for video SEO."""
    tags = []
    
    # Topic keywords
    clean_topic = topic.lower().strip()
    tags.append(clean_topic)
    for word in re.findall(r"\w{3,}", clean_topic):
        if word not in tags:
            tags.append(word)

    # Genre keywords
    tags.append(genre.lower())
    genre_keywords = {
        "anime": ["anime", "anime edit", "anime video", "action anime", "manga", "anime fight"],
        "scifi": ["sci fi", "science fiction", "ai technology", "cyberpunk", "futuristic", "tech"],
        "fantasy": ["fantasy", "magic", "fairy tale", "magical story", "mythical"],
        "documentary": ["documentary", "facts", "history", "educational video", "did you know"],
        "mythology": ["mythology", "ancient history", "sanatan dharma", "indian mythology", "legends"],
        "motivational": ["motivation", "motivational video", "success mindset", "inspiration", "hard work"],
        "drama": ["drama story", "emotional video", "short story", "cinematic", "life lessons"],
        "comedy": ["comedy video", "funny story", "jokes", "humor", "desi comedy"],
        "horror": ["horror story", "scary video", "creepy mystery", "haunted", "suspense"],
    }
    for kw in genre_keywords.get(genre.lower(), [genre.lower()]):
        if kw not in tags:
            tags.append(kw)

    # Headline keywords
    for h in headlines[:5]:
        h_clean = transliterate.deva_to_hinglish(h).lower() if any("\u0900" <= c <= "\u097f" for c in h) else h.lower()
        h_clean = re.sub(r"[^\w\s]", "", h_clean).strip()
        if h_clean and len(h_clean) > 3 and h_clean not in tags:
            tags.append(h_clean)

    # Language tags
    if language in ("hi", "hindi", "hinglish"):
        tags.extend(["hindi", "hinglish", "hindi kahaniya", "hindi stories"])
    else:
        tags.extend(["english", "story in english"])

    # Generic animation / creative tags
    tags.extend(["short film", "2d animation", "storytelling", "viral video"])

    # De-duplicate while preserving order
    seen = set()
    result = []
    for t in tags:
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return result[:25]


# --------------------------------------------------------------------------
# Procedural Metadata Generator (Rule-Based Fallback)
# --------------------------------------------------------------------------

def generate_metadata_procedural(
    script_data: dict,
    timings_data: dict | None = None,
) -> dict:
    """Generate comprehensive, platform-specific metadata procedurally."""
    title = script_data.get("title") or script_data.get("topic") or "Untitled Story"
    topic = script_data.get("topic") or title
    genre = script_data.get("genre") or "scifi"
    language = script_data.get("language") or "hi"
    is_reel = script_data.get("format") in ("reel", "portrait", "square") or script_data.get("aspectRatio") in ("9:16", "4:5", "1:1")
    
    # Extract prose and headlines
    scenes = script_data.get("scenes", [])
    headlines = []
    narration_parts = []
    for s in scenes:
        h = s.get("headline", "")
        if h:
            headlines.append(h)
        for line in s.get("lines", []):
            if line.get("text"):
                narration_parts.append(line["text"])

    full_narration = " ".join(narration_parts)
    chapters = extract_chapters_from_data(script_data, timings_data)
    seo_tags = build_seo_tags(topic, genre, language, headlines)
    category_id = YOUTUBE_CATEGORY_MAP.get(genre.lower(), "24")

    # Transliterated title for international SEO if Devanagari
    has_deva = any("\u0900" <= c <= "\u097f" for c in title)
    hinglish_title = transliterate.deva_to_hinglish(title) if has_deva else title
    
    # 1. YouTube Metadata
    yt_shorts_suffix = " #Shorts" if is_reel else ""
    yt_title = f"{title} | {genre.capitalize()} Story{yt_shorts_suffix}"
    if len(yt_title) > 100:
        yt_title = f"{title[:90]}...{yt_shorts_suffix}"

    chapter_lines = [f"{ts} - {chap}" for ts, chap in chapters]
    chapters_block = "\n".join(chapter_lines)

    yt_hashtags = GENRE_HASHTAG_BANKS.get(genre.lower(), {}).get("youtube", ["#Shorts", "#Viral", "#Trending"])
    if is_reel and "#Shorts" not in yt_hashtags:
        yt_hashtags = ["#Shorts"] + yt_hashtags

    yt_description = (
        f"🔥 {title}\n\n"
        f"Dive into this thrilling {genre} story about {topic}! Watch till the very end to see the shocking twist!\n\n"
        f"⏱️ CHAPTERS & TIMESTAMPS:\n"
        f"{chapters_block}\n\n"
        f"✨ SUMMARY & KEY HIGHLIGHTS:\n"
        f"• Story Topic: {topic.title()}\n"
        f"• Genre: {genre.title()}\n"
        f"• Experience immersive storytelling with rich visuals and atmospheric sound.\n\n"
        f"🔔 SUBSCRIBE for more daily animated stories and exciting videos!\n"
        f"👍 LIKE, COMMENT & SHARE if you enjoyed this video!\n\n"
        f"{' '.join(yt_hashtags[:8])}\n"
    )

    # 2. Instagram Metadata
    ig_hashtags = GENRE_HASHTAG_BANKS.get(genre.lower(), {}).get("instagram", [
        "#reels", "#reelsindia", "#trendingreels", "#viralreels", "#explorepage"
    ])
    ig_hook = f"🚨 {title}!"
    ig_caption = (
        f"{ig_hook}\n\n"
        f"What would you do in this situation? Let us know in the comments below! 👇💬\n\n"
        f"📌 SAVE this reel for later\n"
        f"🚀 SHARE with a friend who loves {genre} stories\n"
        f"👉 FOLLOW for daily captivating stories\n\n"
        f".\n.\n.\n"
        f"{' '.join(ig_hashtags[:28])}"
    )

    # 3. Facebook Metadata
    fb_hashtags = GENRE_HASHTAG_BANKS.get(genre.lower(), {}).get("facebook", ["#Viral", "#Reels", "#Trending"])
    fb_title = f"{title} - Watch Till The End!"
    fb_description = (
        f"Watch what happens in this thrilling {genre} story about {topic}! 🤯\n\n"
        f"Drop your thoughts in the comments! Share with your friends and family. 🚀\n\n"
        f"{' '.join(fb_hashtags)}"
    )

    # 4. TikTok Metadata
    tt_hashtags = GENRE_HASHTAG_BANKS.get(genre.lower(), {}).get("tiktok", ["#fyp", "#viral", "#foryou"])
    tt_caption = f"{title} 😱 Watch till the end! {' '.join(tt_hashtags[:7])}"

    return {
        "topic": topic,
        "genre": genre,
        "language": language,
        "format": script_data.get("format", "reel" if is_reel else "landscape"),
        "aspect_ratio": script_data.get("aspectRatio", "9:16" if is_reel else "16:9"),
        "duration_seconds": round(timings_data.get("totalDuration", 60.0), 2) if timings_data else 60.0,
        "youtube": {
            "title": yt_title,
            "description": yt_description.strip(),
            "tags": seo_tags,
            "category_id": category_id,
            "privacy_status": "public",
            "default_language": language,
            "default_audio_language": language,
            "made_for_kids": False,
            "self_declared_made_for_kids": False,
            "chapters": [{"timestamp": ts, "title": ch} for ts, ch in chapters],
            "hashtags": yt_hashtags[:10],
            "is_shorts": is_reel,
        },
        "instagram": {
            "hook_line": ig_hook,
            "caption": ig_caption.strip(),
            "call_to_action": "Save this reel 📌 | Follow for daily content 🚀",
            "hashtags": ig_hashtags[:30],
            "hashtags_count": min(30, len(ig_hashtags)),
            "recommended_audio": f"Trending {genre.capitalize()} Ambient / Original Audio",
            "cover_timestamp_ms": 1000,
        },
        "facebook": {
            "title": fb_title,
            "description": fb_description.strip(),
            "hashtags": fb_hashtags,
            "tags": seo_tags[:10],
            "call_to_action": "Share with friends & family 👍",
        },
        "tiktok": {
            "title": title,
            "caption": tt_caption.strip(),
            "hashtags": tt_hashtags[:8],
            "sound_prompt": f"Dramatic {genre.capitalize()} Sound / Original Sound",
        },
        "transliterated_title": hinglish_title,
        "summary": f"{genre.title()} story about {topic} ({language})",
        "generated_via": "procedural_rules",
    }


# --------------------------------------------------------------------------
# AI-Driven Metadata Generator (LLM)
# --------------------------------------------------------------------------

def build_ai_metadata_prompt(
    script_data: dict,
    timings_data: dict | None = None,
) -> str:
    """Construct prompt for the LLM to generate high-converting social media metadata."""
    title = script_data.get("title") or script_data.get("topic") or "Untitled Story"
    topic = script_data.get("topic") or title
    genre = script_data.get("genre") or "scifi"
    language = script_data.get("language") or "hi"
    is_reel = script_data.get("format") in ("reel", "portrait", "square") or script_data.get("aspectRatio") in ("9:16", "4:5", "1:1")

    # Extract all narration lines and scene headlines
    scenes = script_data.get("scenes", [])
    scene_descriptions = []
    timing_scenes = timings_data.get("scenes", []) if timings_data else []

    for i, s in enumerate(scenes):
        headline = s.get("headline", f"Scene {i + 1}")
        ts = "00:00"
        if i < len(timing_scenes):
            ts = format_timestamp(timing_scenes[i].get("start", 0.0))
        elif i > 0:
            ts = format_timestamp(i * 15.0)

        lines = [line.get("text", "") for line in s.get("lines", []) if line.get("text")]
        body = " ".join(lines)
        scene_descriptions.append(f"Scene {i + 1} [{ts}] - Headline: {headline}\nNarration: {body}")

    full_script_text = "\n\n".join(scene_descriptions)

    prompt = f"""You are an elite viral social media strategist and YouTube SEO growth expert.
Generate high-converting, viral, platform-optimized upload metadata for this video across YouTube, Instagram, Facebook, and TikTok.

--- VIDEO CONTEXT ---
Title: {title}
Topic: {topic}
Genre: {genre}
Language: {language}
Format: {'Vertical 9:16 Reel/Short' if is_reel else 'Landscape 16:9'}

--- FULL SCRIPT & SCENE TIMESTAMPS ---
{full_script_text}

--- INSTRUCTIONS FOR EACH PLATFORM ---

1. YOUTUBE (Data API v3 & YouTube Studio):
   - title: High CTR, viral SEO title (<90 chars). Include #Shorts if vertical reel.
   - description: Engaging hook in lines 1-2, YouTube chapter timestamps starting from 00:00, 3-4 bullet takeaways, CTA to subscribe/like/share, and 5-8 relevant hashtags at bottom.
   - tags: Array of 20-25 high-search-volume keywords (topic, genre, language, themes).
   - category_id: Choose appropriate YouTube category ("28" for Science/Tech, "27" for Education, "24" for Entertainment, "1" for Animation, "23" for Comedy).

2. INSTAGRAM (Reels & Feed):
   - hook_line: Bold first line to stop the scroll (with emoji).
   - caption: Punchy caption with clean line breaks, emojis, engagement question for comments, and CTA (Save & Share).
   - hashtags: Exactly 25 to 30 strategic hashtags (mix of viral, genre, topic, Hinglish/Hindi, and niche community tags).
   - full_caption: The complete combined text (hook + caption + spacing + all hashtags) ready to copy-paste.

3. FACEBOOK (Reels & Watch):
   - title: Click-worthy title (<200 chars).
   - description: Conversational post text asking an engaging debate/discussion question to maximize shares and comments.
   - hashtags: 3 to 5 targeted hashtags.
   - tags: 8-10 topic tags.

4. TIKTOK:
   - caption: Short, punchy caption (<150 chars) + 6-8 viral hashtags (#fyp, #viral, topic).

--- OUTPUT FORMAT ---
You MUST respond with valid JSON ONLY. Do not include markdown code fences, conversational preamble, or commentary. Use this exact schema:

{{
  "youtube": {{
    "title": "string (<100 chars)",
    "description": "string (with hook, 00:00 timestamps, takeaways, and hashtags)",
    "tags": ["tag1", "tag2", "... (20-25 tags)"],
    "category_id": "string",
    "hashtags": ["#tag1", "#tag2", "..."]
  }},
  "instagram": {{
    "hook_line": "string",
    "caption": "string",
    "call_to_action": "string",
    "hashtags": ["#tag1", "#tag2", "... (25-30 hashtags)"],
    "full_caption": "string"
  }},
  "facebook": {{
    "title": "string",
    "description": "string",
    "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4"],
    "tags": ["tag1", "tag2", "..."],
    "call_to_action": "string"
  }},
  "tiktok": {{
    "title": "string",
    "caption": "string",
    "hashtags": ["#tag1", "#tag2", "..."]
  }},
  "summary": "string"
}}
"""
    return prompt


def generate_metadata_ai(
    script_data: dict,
    timings_data: dict | None = None,
    writer_info: tuple[str, str, str | None, str] | None = None,
) -> dict:
    """Generate upload metadata using LLM with procedural fallback on error."""
    if not writer_info:
        # Fall back to procedural
        return generate_metadata_procedural(script_data, timings_data)

    label, base_url, api_key, model_to_use = writer_info
    prompt = build_ai_metadata_prompt(script_data, timings_data)

    payload = json.dumps(
        {
            "model": model_to_use,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional social media manager and YouTube growth expert. You respond ONLY with valid, raw JSON matching the requested schema.",
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
        "X-Title": "tube2dscript-metadata",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        f"{base_url}/chat/completions", data=payload, headers=headers, method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"].strip()

        # Parse JSON from response
        # Strip potential markdown fences ```json ... ```
        if "```" in content:
            content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
            content = re.sub(r"\s*```$", "", content)

        data = json.loads(content)

        # Merge with structural essentials (chapters, duration, formats)
        procedural_base = generate_metadata_procedural(script_data, timings_data)
        
        # Validate and augment YouTube block
        if "youtube" in data:
            yt = data["youtube"]
            if not yt.get("chapters"):
                yt["chapters"] = procedural_base["youtube"]["chapters"]
            if not yt.get("category_id"):
                yt["category_id"] = procedural_base["youtube"]["category_id"]
            yt["privacy_status"] = "public"
            yt["default_language"] = script_data.get("language", "hi")
            yt["default_audio_language"] = script_data.get("language", "hi")
            yt["made_for_kids"] = False
            yt["self_declared_made_for_kids"] = False
            yt["is_shorts"] = procedural_base["youtube"]["is_shorts"]
            if not yt.get("tags"):
                yt["tags"] = procedural_base["youtube"]["tags"]

        # Validate Instagram block
        if "instagram" in data:
            ig = data["instagram"]
            if not ig.get("hashtags"):
                ig["hashtags"] = procedural_base["instagram"]["hashtags"]
            ig["hashtags_count"] = len(ig.get("hashtags", []))
            ig["recommended_audio"] = procedural_base["instagram"]["recommended_audio"]
            ig["cover_timestamp_ms"] = 1000
            if not ig.get("full_caption"):
                ig["full_caption"] = f"{ig.get('hook_line', '')}\n\n{ig.get('caption', '')}\n\n{' '.join(ig['hashtags'])}"

        # Validate Facebook block
        if "facebook" in data:
            fb = data["facebook"]
            if not fb.get("hashtags"):
                fb["hashtags"] = procedural_base["facebook"]["hashtags"]
            if not fb.get("tags"):
                fb["tags"] = procedural_base["facebook"]["tags"]

        # Validate TikTok block
        if "tiktok" in data:
            tt = data["tiktok"]
            if not tt.get("hashtags"):
                tt["hashtags"] = procedural_base["tiktok"]["hashtags"]

        data["topic"] = procedural_base["topic"]
        data["genre"] = procedural_base["genre"]
        data["language"] = procedural_base["language"]
        data["format"] = procedural_base["format"]
        data["aspect_ratio"] = procedural_base["aspect_ratio"]
        data["duration_seconds"] = procedural_base["duration_seconds"]
        data["transliterated_title"] = procedural_base["transliterated_title"]
        data["generated_via"] = f"ai_llm ({label} - {model_to_use})"

        return data

    except Exception as e:
        print(f"note: AI metadata generation fallback to procedural ({e})", file=sys.stderr)
        return generate_metadata_procedural(script_data, timings_data)


# --------------------------------------------------------------------------
# Main Generation & File Exporters
# --------------------------------------------------------------------------

def generate_metadata(
    script_data: dict,
    timings_data: dict | None = None,
    writer_info: tuple[str, str, str | None, str] | None = None,
    use_ai: bool = True,
) -> dict:
    """Unified entry point for multi-platform metadata generation."""
    if use_ai and writer_info:
        return generate_metadata_ai(script_data, timings_data, writer_info)
    return generate_metadata_procedural(script_data, timings_data)


def format_markdown_report(meta: dict) -> str:
    """Format metadata dictionary into human-readable Markdown for manual upload/copying."""
    yt = meta.get("youtube", {})
    ig = meta.get("instagram", {})
    fb = meta.get("facebook", {})
    tt = meta.get("tiktok", {})

    tags_comma = ", ".join(yt.get("tags", []))
    ig_tags_str = " ".join(ig.get("hashtags", []))
    fb_tags_str = " ".join(fb.get("hashtags", []))
    tt_tags_str = " ".join(tt.get("hashtags", []))

    md = f"""# 🚀 Social Media Upload Metadata

**Topic**: {meta.get('topic')}  
**Genre**: {meta.get('genre', '').title()}  
**Language**: {meta.get('language')}  
**Format**: {meta.get('format')} ({meta.get('aspect_ratio')})  
**Duration**: {meta.get('duration_seconds')}s  
**Generation Engine**: `{meta.get('generated_via', 'procedural')}`  

---

## 📺 1. YouTube (Shorts & Long-Form)

### 📌 Title
```text
{yt.get('title', '')}
```

### 📝 Description (with Timestamps & SEO)
```text
{yt.get('description', '')}
```

### 🏷️ Search Tags (Comma-Separated for YouTube Studio)
```text
{tags_comma}
```

### ⚙️ YouTube Settings
- **Category ID**: `{yt.get('category_id', '24')}`
- **Privacy**: `{yt.get('privacy_status', 'public')}`
- **Made for Kids**: `No`
- **Shorts Optimized**: `{'Yes' if yt.get('is_shorts') else 'No'}`

---

## 📸 2. Instagram (Reels & Feed)

### 🎯 Hook Headline
> {ig.get('hook_line', '')}

### 📋 Full Caption & Hashtags (Ready to Copy-Paste)
```text
{ig.get('full_caption', ig.get('caption', ''))}
```

### #️⃣ Hashtags ({ig.get('hashtags_count', len(ig.get('hashtags', [])))} tags)
```text
{ig_tags_str}
```

### 🎵 Audio & Cover
- **Recommended Audio**: {ig.get('recommended_audio', 'Original Audio')}
- **Cover Frame**: {ig.get('cover_timestamp_ms', 1000)} ms

---

## 👥 3. Facebook (Watch & Reels)

### 📌 Title
```text
{fb.get('title', '')}
```

### 📝 Description
```text
{fb.get('description', '')}
```

### #️⃣ Hashtags
```text
{fb_tags_str}
```

---

## 🎵 4. TikTok

### 📝 Caption
```text
{tt.get('caption', '')}
```

### #️⃣ Hashtags
```text
{tt_tags_str}
```

---
*Generated automatically by Tube2DScript Multi-Platform Metadata Engine.*
"""
    return md


def export_metadata(
    meta: dict,
    out_dir_or_file: Path | str,
    base_name: str = "metadata",
) -> tuple[Path, Path]:
    """Export metadata to both machine-readable JSON and formatted Markdown.

    Returns (json_path, markdown_path).
    """
    target = Path(out_dir_or_file)
    if target.suffix == ".json":
        json_path = target
        md_path = target.with_suffix(".md")
    elif target.suffix == ".md":
        md_path = target
        json_path = target.with_suffix(".json")
    else:
        # Directory
        target.mkdir(parents=True, exist_ok=True)
        json_path = target / f"{base_name}.json"
        md_path = target / f"{base_name}.md"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(format_markdown_report(meta), encoding="utf-8")

    return json_path, md_path


# --------------------------------------------------------------------------
# Standalone CLI Interface
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--script", default="script.json", help="path to script.json (default: script.json)")
    parser.add_argument("--project", default="src/generated/project.json", help="path to project.json with timings")
    parser.add_argument("--out", default="out", help="output directory or file path for metadata (default: out)")
    parser.add_argument("--name", default="metadata", help="base filename (default: metadata)")
    parser.add_argument("--no-ai", action="store_true", help="force procedural rule-based generation (zero API calls)")
    parser.add_argument("--model", help="LLM model override for AI metadata generation")
    args = parser.parse_args()

    script_p = Path(args.script)
    if not script_p.exists():
        print(f"error: script file {script_p} not found", file=sys.stderr)
        return 1

    script_data = json.loads(script_p.read_text(encoding="utf-8"))

    project_p = Path(args.project)
    timings_data = None
    if project_p.exists():
        try:
            proj = json.loads(project_p.read_text(encoding="utf-8"))
            timings_data = proj.get("timings")
        except Exception:
            pass

    import new_video  # noqa: E402
    writer_info = None if args.no_ai else new_video.resolve_llm_writer(requested_model=args.model)

    print("=" * 74)
    print("GENERATING MULTI-PLATFORM SOCIAL MEDIA METADATA")
    print("=" * 74)
    print(f"  topic     : {script_data.get('topic')}")
    print(f"  genre     : {script_data.get('genre')}")
    print(f"  language  : {script_data.get('language')}")
    print(f"  engine    : {'AI LLM (' + writer_info[0] + ')' if writer_info and not args.no_ai else 'Procedural Rules'}")

    meta = generate_metadata(
        script_data,
        timings_data=timings_data,
        writer_info=writer_info,
        use_ai=not args.no_ai,
    )

    json_p, md_p = export_metadata(meta, args.out, base_name=args.name)
    print()
    print(f"  ✓ Saved machine-readable JSON : {json_p}")
    print(f"  ✓ Saved formatted Markdown     : {md_p}")
    print()
    print("=" * 74)
    print("YOUTUBE METADATA PREVIEW")
    print("=" * 74)
    print(f"  Title: {meta['youtube']['title']}")
    print(f"  Category ID: {meta['youtube']['category_id']} | Tags: {len(meta['youtube']['tags'])} keywords")
    print(f"  Chapters: {len(meta['youtube']['chapters'])} timestamps")
    print()
    print("=" * 74)
    print("INSTAGRAM REEL CAPTION PREVIEW")
    print("=" * 74)
    print(meta["instagram"]["hook_line"])
    print(f"Hashtags: {meta['instagram']['hashtags_count']} curated tags")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
