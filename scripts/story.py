#!/usr/bin/env python3
"""Blueprint-first story writing: one story bible, then chunked continuation.

Why this exists
---------------
Asking a model for "24 scenes, about 1800 words" in a single call gives a model
that invents the plot as it types. Scene 14 has no idea what scene 3 established,
so what comes back reads as fragments cut out of some larger story. A long
response also gets truncated, and because prose used to be bound to story duties
by position, one dropped scene shifted every scene after it.

The shape here is:

    plan -> bible (one call) -> chunk 1 -> chunk 2 -> ... -> validate -> repair

The bible is written first and shown to every later call, so the writer knows how
long the story is, what happens in every scene, and where it currently stands.
Each chunk receives the tail of the previous chunk verbatim, a one-line summary of
everything already written, and its own absolute scene range. Scene numbers stay
absolute throughout, so prose is bound by index instead of by position.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

__all__ = [
    "Issue",
    "WrittenScene",
    "BibleWriter",
    "bible_ok",
    "build_bible_prompt",
    "build_chunk_prompt",
    "build_repair_prompt",
    "chunk_ranges",
    "coerce_blocks",
    "is_conversational_chatter",
    "normalize_bible",
    "parse_bible",
    "parse_blocks",
    "plan_chunks",
    "scene_duties",
    "summarise_written",
    "validate_script",
    "written_tail",
]


# ---------------------------------------------------------------------------
# Chatter detection
# ---------------------------------------------------------------------------
# A refusal or a "how can I help you today" is not narration. Detecting it costs
# nothing and saves a whole run.

CHATTER_PATTERNS: tuple[str, ...] = (
    "looks like your message came through blank",
    "how can i help you today",
    "i am an ai language model",
    "as an ai language model",
    "i cannot fulfill this request",
    "i can't fulfill this request",
    "is there anything else i can help you with",
    "it appears your message is empty",
    "it seems your prompt is empty",
    "your message was blank",
    "सहायता कर सकता हूँ",
    "किस प्रकार सहायता",
    "कैसे मदद कर सकता हूँ",
    "नमस्ते! आज मैं",
    "कृपया अपने प्रश्न या विषय",
    "आपकी क्या सहायता",
    "मैं एक एआई",
    "मैं एक ai",
    "मैं आपकी क्या मदद",
)


def is_conversational_chatter(text: str) -> bool:
    """True when a model returned chatter/refusal rather than narration."""
    lowered = (text or "").lower().strip()
    return any(pattern in lowered for pattern in CHATTER_PATTERNS)


# ---------------------------------------------------------------------------
# Written scenes
# ---------------------------------------------------------------------------


@dataclass
class WrittenScene:
    """One scene of prose, bound to an absolute scene number when we know it.

    `n` is the 1-based scene number the writer labelled the block with. It is
    None only for legacy / hand-written input that carries no numbers, in which
    case the caller falls back to positional binding.
    """

    n: int | None = None
    headline: str = ""
    visual: str = ""
    body: str = ""

    @property
    def words(self) -> int:
        return len(self.body.split())

    @property
    def is_empty(self) -> bool:
        return not self.body.strip()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_RE_DIVIDER = (
    re.compile(r"^\s*#{1,4}\s*(\d+)"),
    re.compile(r"^\s*scene\s*(\d+)\s*[|:.\-]", re.IGNORECASE),
    re.compile(r"^\s*(\d+)\s*\|\s*"),
)
_RE_HEADLINE = re.compile(
    r"^\s*(?:#{1,4}\s*)?(?:scene\s*)?(\d+)\s*[|:.\-]?\s*(.*)$", re.IGNORECASE
)


def _divider_index(line: str) -> int | None:
    """The scene number if this line opens a scene block, else None."""
    for pattern in _RE_DIVIDER:
        match = pattern.match(line)
        if match:
            return int(match.group(1))
    return None


def _split_visual(headline: str) -> tuple[str, str]:
    """Pull an inline visual tag out of a headline. Returns (headline, visual)."""
    if " | visual:" in headline.lower():
        parts = re.split(r"\s*\|\s*visual:\s*", headline, flags=re.IGNORECASE, maxsplit=1)
        return parts[0].strip(), (parts[1].strip() if len(parts) > 1 else "")
    if "[visual:" in headline.lower():
        match = re.search(r"\[visual:\s*([^\]]+)\]", headline, flags=re.IGNORECASE)
        if match:
            return re.sub(r"\[visual:\s*[^\]]+\]", "", headline, flags=re.IGNORECASE).strip(), match.group(1).strip()
    if "(visual:" in headline.lower():
        match = re.search(r"\(visual:\s*([^\)]+)\)", headline, flags=re.IGNORECASE)
        if match:
            return re.sub(r"\(visual:\s*[^\)]+\)", "", headline, flags=re.IGNORECASE).strip(), match.group(1).strip()
    return headline, ""


def _clean_body(lines: Sequence[str], visual: str) -> tuple[str, str]:
    """Join spoken lines, strip visual tags from the prose, fix trailing punctuation."""
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^\[?\s*visual\s*:\s*", stripped, flags=re.IGNORECASE):
            if not visual:
                visual = re.sub(r"^\[?\s*visual\s*:\s*", "", stripped, flags=re.IGNORECASE).rstrip("]").strip()
            continue
        cleaned = re.sub(r"\[visual:\s*[^\]]+\]", "", line, flags=re.IGNORECASE)
        cleaned = re.sub(r"\(visual:\s*[^\)]+\)", "", cleaned, flags=re.IGNORECASE)
        kept.append(cleaned)

    body = " ".join(" ".join(kept).split())
    if not body:
        return "", visual
    # A scene ending on a dangling comma reads as a scene cut in half, which is
    # exactly the artefact this module exists to remove.
    body = body.rstrip(" ,-\u2013\u2014;:")
    if body and not body.endswith(("\u0964", ".", "!", "?", '"', "\u201d", "'")):
        body += "\u0964" if re.search(r"[\u0900-\u097f]", body) else "."
    return body, visual


def parse_blocks(text: str, scene_count: int = 999) -> tuple[str, list[WrittenScene]]:
    """Parse `TITLE:` plus `###N|headline | VISUAL: tags` blocks.

    Returns (title, scenes). Each scene carries the absolute number the writer
    used, which is what lets the caller bind prose by index.
    """
    if is_conversational_chatter(text):
        return "", []

    title = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("TITLE:"):
            title = stripped.split(":", 1)[1].strip()
            break

    raw: list[tuple[int | None, str, list[str]]] = []
    current_n: int | None = None
    current_headline = ""
    buffer: list[str] = []
    started = False

    def flush() -> None:
        if started:
            raw.append((current_n, current_headline, list(buffer)))

    for line in text.splitlines():
        index = _divider_index(line)
        if index is not None:
            flush()
            buffer = []
            match = _RE_HEADLINE.match(line)
            current_headline = match.group(2).strip() if match else ""
            current_n = index
            started = True
            continue
        if started:
            buffer.append(line)
    flush()

    if not raw:
        # The model ignored the format. Split on blank lines so a sloppy response
        # is still usable, but never invent more scenes than the plan has room for:
        # extra paragraphs are what used to shove every real scene out of place.
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        usable = [p for p in paragraphs if not is_conversational_chatter(p) and len(p.split()) > 3]
        raw = [(None, "", [p]) for p in usable[: max(1, scene_count)]]

    scenes: list[WrittenScene] = []
    for number, headline, lines in raw:
        headline, visual = _split_visual(headline)
        body, visual = _clean_body(lines, visual)
        if body and not is_conversational_chatter(body):
            scenes.append(WrittenScene(n=number, headline=headline, visual=visual, body=body))

    if scene_count < 999:
        # Keep only in-range scenes, in order, without duplicating an index.
        seen: set[int] = set()
        trimmed: list[WrittenScene] = []
        for scene in scenes:
            if scene.n is not None:
                if scene.n < 1 or scene.n > scene_count or scene.n in seen:
                    continue
                seen.add(scene.n)
            trimmed.append(scene)
        scenes = trimmed

    return title, scenes


def coerce_blocks(blocks: Sequence[Any]) -> list[WrittenScene]:
    """Accept anything a caller might hand us and return WrittenScene objects.

    Legacy shapes stay supported so `--narration` files and older callers keep
    working: (headline, body) and (headline, visual, body) tuples, plus the new
    (n, headline, visual, body) form and plain dicts.
    """
    out: list[WrittenScene] = []
    for item in blocks:
        if isinstance(item, WrittenScene):
            out.append(item)
            continue
        if isinstance(item, dict):
            out.append(
                WrittenScene(
                    n=item.get("n") or item.get("scene") or item.get("index"),
                    headline=str(item.get("headline") or ""),
                    visual=str(item.get("visual") or item.get("visualKeywords") or ""),
                    body=str(item.get("body") or item.get("narration") or item.get("text") or ""),
                )
            )
            continue
        if not isinstance(item, (tuple, list)):
            continue
        parts = list(item)
        if len(parts) >= 4 and str(parts[0]).strip().isdigit():
            out.append(WrittenScene(n=int(str(parts[0]).strip()), headline=str(parts[1]), visual=str(parts[2]), body=str(parts[3])))
        elif len(parts) == 3:
            headline, visual = _split_visual(str(parts[0]))
            out.append(WrittenScene(n=None, headline=headline, visual=visual or str(parts[1]), body=str(parts[2])))
        elif len(parts) == 2:
            headline, visual = _split_visual(str(parts[0]))
            out.append(WrittenScene(n=None, headline=headline, visual=visual, body=str(parts[1])))
        elif len(parts) == 1:
            out.append(WrittenScene(n=None, headline="", visual="", body=str(parts[0])))
    return [scene for scene in out if not scene.is_empty]


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------


def scene_duties(planned: dict, bible: dict | None = None) -> list[dict[str, Any]]:
    """One dict per planned scene: number, movement, duty, target words, bible notes."""
    beats_by_number: dict[int, dict] = {}
    if bible:
        for entry in bible.get("sceneBeats") or []:
            try:
                beats_by_number[int(entry.get("n"))] = entry
            except (TypeError, ValueError):
                continue

    duties: list[dict[str, Any]] = []
    for index, scene in enumerate(planned.get("scenes") or [], start=1):
        note = beats_by_number.get(index) or {}
        duties.append(
            {
                "n": index,
                "movement": scene.get("_story", ""),
                "beat": scene.get("beat", ""),
                "duty": scene.get("_storyNote", ""),
                "words": int(scene.get("_targetWords") or planned.get("target", {}).get("wordsPerScene") or 40),
                "happens": str(note.get("whatHappens") or "").strip(),
                "ends_on": str(note.get("endsOn") or "").strip(),
                "continues_from": note.get("continuesFrom"),
                "loops": note.get("openLoops") or [],
            }
        )
    return duties


def written_tail(written: Sequence[WrittenScene], count: int = 2) -> str:
    """The last few scenes verbatim, in the exact format the writer must continue in."""
    tail = [scene for scene in written if not scene.is_empty][-max(0, count):]
    chunks: list[str] = []
    for scene in tail:
        number = scene.n if scene.n is not None else "?"
        visual = f" | VISUAL: {scene.visual}" if scene.visual else ""
        chunks.append(f"###{number}|{scene.headline}{visual}\n{scene.body}")
    return "\n\n".join(chunks)


def summarise_written(written: Sequence[WrittenScene], lead_words: int = 12, max_lines: int = 40) -> str:
    """One line per scene already written: cheap, deterministic, no extra call."""
    lines: list[str] = []
    for scene in written:
        if scene.is_empty:
            continue
        number = scene.n if scene.n is not None else len(lines) + 1
        lead = " ".join(scene.body.split()[:lead_words])
        lines.append(f"  scene {number} ({scene.words}w): {lead}...")
    if len(lines) > max_lines:
        head = lines[: max_lines // 2]
        tail = lines[-max_lines // 2:]
        return "\n".join(head + [f"  ... {len(lines) - max_lines} scenes omitted ..."] + tail)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The story bible
# ---------------------------------------------------------------------------

BIBLE_SYSTEM_PROMPT = (
    "You are a story architect for narrated video. You reply with exactly one JSON object "
    "and nothing else: no prose, no markdown fences, no commentary."
)

_BIBLE_KEYS = (
    "title",
    "logline",
    "protagonist",
    "setting",
    "span",
    "arc",
    "sceneBeats",
    "payoff",
    "banned",
)


def build_bible_prompt(
    planned: dict,
    genre_profile: Any = None,
    research: dict | None = None,
    *,
    is_hindi: bool = True,
) -> str:
    """The one call that decides what the story is, before any prose is written."""
    target = planned.get("target") or {}
    scene_count = int(target.get("scenes") or len(planned.get("scenes") or []))
    minutes = float(target.get("minutes") or 1.0)
    total_words = int(target.get("totalWords") or 0)

    lines: list[str] = [
        "Design ONE complete story and return it as a single JSON object.",
        "",
        f"Topic        : {planned.get('topic')}",
        f"Genre        : {getattr(genre_profile, 'name', planned.get('genre', ''))}",
        f"Tone         : {getattr(genre_profile, 'tone', '')}",
        f"Language     : {'Hindi (Devanagari, JSON string values may be Hindi)' if is_hindi else 'English'}",
        f"Total length : {minutes:g} minutes ({scene_count} scenes, about {total_words} spoken words)",
        "",
        "THE LENGTH IS FIXED. This story occupies exactly "
        f"{scene_count} scenes and about {total_words} words. Do not plan a longer or "
        "shorter story, and do not plan a story that would need a sequel.",
        "",
        "Every scene in `sceneBeats` must have a job, and the chain of jobs must form one "
        "continuous story with a beginning, an escalation and an ending inside these scenes.",
        "",
    ]

    if research:
        concepts = ((research.get("ai_synthesis") or {}).get("concepts")) or []
        seo_tags = ((research.get("ai_synthesis") or {}).get("seo_tags")) or []
        search_terms: list[str] = []
        clusters = research.get("keywords") or {}
        for key in ("youtube_trending", "questions_and_curiosity", "long_tail_keywords"):
            for item in clusters.get(key) or []:
                text = str(item).strip()
                if text and text not in search_terms:
                    search_terms.append(text)
        if concepts or search_terms or seo_tags:
            lines.append("LIVE SEARCH DEMAND (steer the story with this; never state it on air):")
            for index, concept in enumerate(concepts[:3], start=1):
                title = str(concept.get("title") or "").strip()
                hook = str(concept.get("hook") or "").strip()
                if title:
                    lines.append(f"  {index}. {title}" + (f" - {hook}" if hook else ""))
            if search_terms:
                lines.append("  What viewers type: " + ", ".join(search_terms[:10]))
            if seo_tags:
                lines.append("  Echo in the title: " + ", ".join(str(tag) for tag in seo_tags[:6]))
            lines.append("")

    lines += [
        "Return this exact shape:",
        "",
        json.dumps(
            {
                "title": "short viral title",
                "logline": "one sentence: who wants what, and what stands in the way",
                "protagonist": {"name": "", "want": "", "flaw": ""},
                "setting": {"place": "", "era": ""},
                "span": {"totalScenes": scene_count, "totalWords": total_words, "runtimeMinutes": minutes},
                "arc": [
                    {
                        "movement": "hook|setup|inciting|rising|climax|resolution|button",
                        "scenes": [1, 2],
                        "summary": "what this movement does to the story",
                        "endsOn": "the state the story is in when the movement closes",
                    }
                ],
                "sceneBeats": [
                    {
                        "n": 1,
                        "beat": "establish|dread|approach|reveal|react|resolve",
                        "whatHappens": "the concrete event in this scene",
                        "continuesFrom": None,
                        "endsOn": "the hook or question this scene hands to the next one",
                        "openLoops": ["thread opened and not yet paid off"],
                        "words": 44,
                    }
                ],
                "payoff": {"scene": scene_count, "resolves": ["every loop that must close"]},
                "banned": ["phrases or habits this particular story must avoid"],
            },
            ensure_ascii=False,
            indent=1,
        ),
        "",
        f"`sceneBeats` must contain exactly {scene_count} entries numbered 1..{scene_count}, in order,",
        f"with the words column summing to roughly {total_words}. Nothing may be resolved before the",
        "final scene, and the final scene must pay off every open loop.",
    ]
    return "\n".join(lines)


def parse_bible(text: str) -> dict | None:
    """Pull one JSON object out of a model response, fences and prose and all."""
    if not text:
        return None
    body = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", body, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        body = fence.group(1).strip()

    try:
        parsed = json.loads(body)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        pass

    # Fall back to the outermost balanced braces.
    start = body.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for offset, char in enumerate(body[start:], start=start):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(body[start : offset + 1])
                    except (ValueError, TypeError):
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
        start = body.find("{", start + 1)
    return None


def bible_ok(bible: dict | None, planned: dict) -> list[str]:
    """Problems that mean the bible is not worth writing prose from."""
    if not isinstance(bible, dict):
        return ["no JSON object parsed"]
    problems: list[str] = []
    if not str(bible.get("logline") or "").strip():
        problems.append("no logline")
    if not isinstance(bible.get("arc"), list) or not bible.get("arc"):
        problems.append("no arc")

    entries = bible.get("sceneBeats")
    if not isinstance(entries, list) or not entries:
        problems.append("no sceneBeats")
    else:
        planned_count = len(planned.get("scenes") or [])
        seen: set[int] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                seen.add(int(entry.get("n")))
            except (TypeError, ValueError):
                continue
        if planned_count:
            coverage = len(seen & set(range(1, planned_count + 1))) / planned_count
            if coverage < 0.6:
                problems.append(f"sceneBeats cover only {coverage:.0%} of the {planned_count} planned scenes")
    return problems


def normalize_bible(bible: dict, planned: dict, *, is_hindi: bool = True) -> dict:
    """Fill any gaps from the planner so later prompts never see a hole.

    A model that returns 21 of 24 beats, or forgets a movement, still produces a
    usable bible. The planner already knows each scene's duty, so gaps get filled
    from it rather than triggering another paid call.
    """
    target = planned.get("target") or {}
    scene_count = int(target.get("scenes") or len(planned.get("scenes") or []))
    duties = scene_duties(planned)

    raw_beats = {
        int(entry.get("n")): entry
        for entry in (bible.get("sceneBeats") or [])
        if isinstance(entry, dict) and str(entry.get("n")).lstrip("-").isdigit()
    }

    scene_beats: list[dict[str, Any]] = []
    for duty in duties:
        number = duty["n"]
        entry = dict(raw_beats.get(number) or {})
        entry["n"] = number
        entry.setdefault("beat", duty["beat"])
        if not str(entry.get("whatHappens") or "").strip():
            entry["whatHappens"] = duty["duty"]
        if not str(entry.get("endsOn") or "").strip():
            entry["endsOn"] = duty["duty"]
        if entry.get("continuesFrom") is None and number > 1:
            entry["continuesFrom"] = number - 1
        if not isinstance(entry.get("openLoops"), list):
            entry["openLoops"] = []
        try:
            entry["words"] = int(entry.get("words") or duty["words"])
        except (TypeError, ValueError):
            entry["words"] = duty["words"]
        scene_beats.append(entry)

    bible = dict(bible)
    bible["sceneBeats"] = scene_beats
    bible["span"] = {
        "totalScenes": scene_count,
        "totalWords": int(target.get("totalWords") or 0),
        "runtimeMinutes": float(target.get("minutes") or 0.0),
    }
    if not isinstance(bible.get("arc"), list) or not bible.get("arc"):
        bible["arc"] = []
    bible.setdefault("banned", [])
    bible.setdefault("language", "hi" if is_hindi else "en")
    return bible


# ---------------------------------------------------------------------------
# Chunk planning
# ---------------------------------------------------------------------------

# Around 450 output words per call sits comfortably inside every provider cap and
# keeps the model's attention on the scenes in front of it.
CHUNK_SCENES = 8


def plan_chunks(scene_count: int, chunk_scenes: int = CHUNK_SCENES) -> list[tuple[int, int]]:
    """Split 1..scene_count into contiguous (first, last) inclusive ranges."""
    if scene_count <= 0:
        return []
    size = max(1, int(chunk_scenes))
    if size >= scene_count:
        return [(1, scene_count)]
    return [(start, min(start + size - 1, scene_count)) for start in range(1, scene_count + 1, size)]


def chunk_ranges(scene_count: int, chunk_scenes: int = CHUNK_SCENES) -> list[tuple[int, int]]:
    """Alias kept for readability at call sites."""
    return plan_chunks(scene_count, chunk_scenes)


# ---------------------------------------------------------------------------
# Chunk prompt
# ---------------------------------------------------------------------------


def build_chunk_prompt(
    planned: dict,
    bible: dict,
    chunk: tuple[int, int],
    written: Sequence[WrittenScene],
    *,
    genre_profile: Any = None,
    research: dict | None = None,
    is_hindi: bool = True,
    keep_index: bool = True,
) -> str:
    """The continuation brief: full bible, what is already written, then this chunk."""
    first, last = chunk
    target = planned.get("target") or {}
    total_scenes = int(target.get("scenes") or len(planned.get("scenes") or []))
    total_words = int(target.get("totalWords") or 0)
    done_words = sum(scene.words for scene in written)
    is_first = first == 1
    is_last = last >= total_scenes

    duties = {duty["n"]: duty for duty in scene_duties(planned, bible)}

    lines: list[str] = []
    lines.append(
        f"You are writing the spoken narration for scenes {first}-{last} of {total_scenes} "
        "of ONE single continuous story."
    )
    lines.append("")

    if is_hindi:
        lines += [
            "### CRITICAL LANGUAGE MANDATE: 100% SPOKEN HINDI IN DEVANAGARI (\u0939\u093f\u0902\u0926\u0940 \u0932\u093f\u092a\u093f)",
            "- Every spoken sentence must be conversational spoken Hindi in Devanagari script.",
            "- English loanwords go in Devanagari (\u092b\u094b\u0928, \u0939\u0949\u0938\u094d\u092a\u093f\u091f\u0932, \u0938\u094d\u091f\u0947\u0936\u0928). Never write an English sentence.",
            "- Avoid archaic Sanskritised words (\u091a\u0932 \u0926\u0942\u0930\u092d\u093e\u0937, \u0938\u0902\u0917\u0923\u0915, \u091a\u093f\u0915\u093f\u0924\u094d\u0938\u093e\u0932\u092f). Use the word a real speaker uses.",
            "",
        ]

    lines += [
        "=== STORY BIBLE (already decided; do not change it) ===",
        json.dumps(bible, ensure_ascii=False, indent=1),
        "",
    ]

    if written:
        lines += [
            "=== ALREADY WRITTEN (never repeat or re-introduce any of this) ===",
            summarise_written(written),
            "",
            "=== LAST SCENE WRITTEN, VERBATIM (continue from where this stops) ===",
            written_tail(written, 2),
            "",
        ]

    lines += [
        f"=== WRITE SCENES {first} TO {last} NOW ===",
    ]
    for number in range(first, last + 1):
        duty = duties.get(number, {})
        detail = f"  scene {number} (~{duty.get('words', 40)} words | {duty.get('beat', '')}): {duty.get('duty', '')}"
        if duty.get("happens"):
            detail += f"\n      happens: {duty['happens']}"
        if duty.get("ends_on"):
            detail += f"\n      hands off: {duty['ends_on']}"
        if duty.get("loops"):
            detail += f"\n      open loops: {', '.join(str(loop) for loop in duty['loops'])}"
        lines.append(detail)
    lines.append("")

    lines += [
        "=== POSITION CONTRACT (this is what keeps the story continuous) ===",
        f"- You are writing scenes {first}-{last} out of {total_scenes}. The finished story is about {total_words} words; about {done_words} are written so far.",
    ]
    if is_first:
        lines.append(
            "- Scene 1 must open the story in its first 10-12 words. No greeting, no preamble, no background lecture."
        )
    else:
        lines.append(
            f"- Do NOT restart. Do NOT re-introduce any character, place or object already established. Scene {first} must begin mid-motion, continuing directly from scene {first - 1}."
        )
    if is_last:
        lines.append(
            "- This is the final chunk: land the climax and pay off every open loop listed in the bible. Close the story. No cliffhanger, no 'to be continued'."
        )
    else:
        lines.append(
            f"- Do NOT summarise and do NOT conclude. Scene {last} must end on the hand-off listed for it, leaving the story unfinished and pulling forward."
        )
    lines += [
        "- Plain spoken prose only: no markdown, no emoji, no stage directions, no sound-effect labels, no scene numbers inside the prose.",
        "- Write numbers as words ('\u0926\u094b', not '2') because TTS reads exactly what is written.",
        "- Never mention cameras, videos, or the audience.",
        "",
        "=== OUTPUT FORMAT, exactly this and nothing else ===",
    ]

    for number in range(first, last + 1):
        duty = duties.get(number, {})
        lines.append(f"###{number}|<short punchy scene headline> | VISUAL: <3-5 concrete English visual keywords>")
        lines.append(f"<{duty.get('words', 40)} words of spoken narration for scene {number}>")
        lines.append("")
    lines.append("Do not renumber. Use the absolute scene numbers shown above.")
    return "\n".join(lines)


def build_repair_prompt(
    planned: dict,
    bible: dict | None,
    issue: "Issue",
    written: Sequence[WrittenScene],
    *,
    is_hindi: bool = True,
) -> str:
    """Rewrite exactly one broken scene, with its neighbours for continuity."""
    by_number: dict[int, WrittenScene] = {
        scene.n: scene for scene in written if scene.n is not None
    }
    number = issue.scene
    duties = {duty["n"]: duty for duty in scene_duties(planned, bible)}
    duty = duties.get(number, {})
    before = by_number.get(number - 1)
    after = by_number.get(number + 1)
    draft = by_number.get(number)

    lines: list[str] = [
        f"Rewrite ONLY scene {number} of this story. Keep every other scene exactly as it is.",
        f"Problem with the current draft: {issue.detail}",
        "",
        f"Scene {number} must do this: {duty.get('duty', '')}",
    ]
    if duty.get("happens"):
        lines.append(f"It must contain: {duty['happens']}")
    if duty.get("ends_on"):
        lines.append(f"It must hand off: {duty['ends_on']}")
    lines += [f"Length: about {duty.get('words', 40)} words.", ""]

    if bible:
        lines += ["Story context (do not contradict):", json.dumps(bible, ensure_ascii=False), ""]
    if before and not before.is_empty:
        lines += [f"Previous scene (scene {number - 1}) ended like this:", before.body[-400:], ""]
    if after and not after.is_empty:
        lines += [f"Next scene (scene {number + 1}) begins like this:", after.body[:400], ""]
    if draft and not draft.is_empty:
        lines += ["The draft to replace:", draft.body, ""]
    if is_hindi:
        lines += ["Write in conversational Hindi, Devanagari script only.", ""]
    lines += [
        "Reply with exactly two lines and nothing else:",
        f"###{number}|<scene headline> | VISUAL: <3-5 English visual keywords>",
        f"<rewritten {duty.get('words', 40)}-word narration for scene {number}>",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Deliberately wide: a scene 35% off its word budget still sounds fine, and
# repairing it costs a paid call.
WORD_TOLERANCE = 0.35

_RESTART_MARKERS = (
    "\u091c\u0948\u0938\u093e \u0915\u093f \u0939\u092e\u0928\u0947 \u092a\u0939\u0932\u0947 \u092c\u0924\u093e\u092f\u093e",
    "\u092a\u0939\u0932\u0947 \u0939\u0940 \u092c\u0924\u093e\u092f\u093e",
    "\u0905\u092c \u0924\u0915 \u0939\u092e\u0928\u0947 \u0926\u0947\u0916\u093e",
    "\u0938\u093e\u0930\u093e\u0902\u0936",
    "\u0915\u094d\u0930\u092e\u0936\u0903",
    "\u0905\u0917\u0932\u0947 \u092d\u093e\u0917 \u092e\u0947\u0902",
    "to be continued",
    "as we discussed earlier",
    "as mentioned earlier",
    "previously on",
    "in summary",
    "to summarise",
    "to summarize",
)

_DRIFT_WORDS = 45  # first-sentence overlap above this reads as a recap


@dataclass
class Issue:
    """One problem with one scene, and what to do about it."""

    scene: int
    kind: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"scene {self.scene}: {self.kind} ({self.detail})"


def _first_words(text: str, count: int = 4) -> list[str]:
    return [word.strip("\u0964.,!?\"'()") for word in text.split()[:count] if word.strip()]


def validate_script(
    written: Sequence[WrittenScene],
    planned: dict,
    bible: dict | None = None,
) -> list[Issue]:
    """Everything wrong with the prose, scene by scene. Never raises."""
    expected = len(planned.get("scenes") or [])
    duties = {duty["n"]: duty for duty in scene_duties(planned, bible)}
    by_number: dict[int, WrittenScene] = {}
    issues: list[Issue] = []

    for scene in written:
        if scene.n is None:
            continue
        if scene.n in by_number:
            issues.append(Issue(scene.n, "duplicate", "scene number appears twice"))
        else:
            by_number[scene.n] = scene

    for number in range(1, expected + 1):
        scene = by_number.get(number)
        duty = duties.get(number, {})
        target = int(duty.get("words") or 40)

        if scene is None or scene.is_empty:
            issues.append(Issue(number, "missing", "no narration was written"))
            continue

        if target > 8:
            drift = abs(scene.words - target) / target
            if drift > WORD_TOLERANCE:
                issues.append(
                    Issue(number, "length", f"{scene.words} words against a target of {target} ({drift:.0%} off)")
                )

        lowered = scene.body.lower()
        for marker in _RESTART_MARKERS:
            if marker in lowered:
                issues.append(Issue(number, "restart", f"recap / continuation marker {marker!r}"))
                break

        previous = by_number.get(number - 1)
        if previous is not None and not previous.is_empty and scene.words >= 8:
            # Same opening and same length is a copy: a real continuation rarely
            # repeats eight words verbatim, and every scene shares a word target.
            lead = _first_words(scene.body, 8)
            if lead and lead == _first_words(previous.body, 8) and scene.words == previous.words:
                issues.append(Issue(number, "duplicate", "scene reads as a copy of the previous scene"))

    if expected and all(by_number.get(number) is None for number in range(1, expected + 1)):
        issues.append(Issue(0, "empty", "the writer produced nothing usable"))

    return issues


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

BibleWriter = Callable[..., str]


def build_bible(
    planned: dict,
    llm: BibleWriter | None,
    *,
    genre_profile: Any = None,
    research: dict | None = None,
    is_hindi: bool = True,
    attempts: int = 3,
    prompt_override: str | None = None,
    verbose: bool = True,
) -> tuple[dict | None, dict[str, Any]]:
    """Write the story bible, retrying until it is usable.

    Returns (bible, report). The bible is None only when the model never produced
    a usable one; the caller then falls back to writing without a blueprint.
    """
    draft = prompt_override or build_bible_prompt(planned, genre_profile, research, is_hindi=is_hindi)
    report: dict[str, Any] = {"attempts": [], "prompt": draft, "accepted": False, "problems": []}

    if llm is None:
        report["problems"] = ["no writer available"]
        return None, report

    for attempt in range(1, max(1, attempts) + 1):
        try:
            response = llm(draft, temperature=0.7, max_tokens=4096)
        except Exception as exc:  # noqa: BLE001 - one bad attempt is not fatal
            report["attempts"].append({"attempt": attempt, "error": f"{type(exc).__name__}: {exc}"})
            if verbose:
                print(f"  [bible {attempt}] failed: {type(exc).__name__}: {exc}")
            continue

        parsed = parse_bible(response)
        problems = bible_ok(parsed, planned)
        report["attempts"].append(
            {"attempt": attempt, "parsed": parsed is not None, "problems": problems, "raw": response}
        )
        if parsed is not None and not problems:
            report["accepted"] = True
            report["problems"] = []
            return normalize_bible(parsed, planned, is_hindi=is_hindi), report

        report["problems"] = problems
        if verbose:
            print(f"  [bible {attempt}] rejected: {', '.join(problems) or 'unparseable JSON'}")

    # Last resort: a bible built purely from the planner's own duties. It carries
    # no story, but it does carry the true length and per-scene jobs, which is
    # what the chunk prompt needs to stay continuous.
    report["problems"] = report["problems"] or ["bible fell back to the planner's scene duties"]
    report["fallback"] = True
    return normalize_bible({}, planned, is_hindi=is_hindi), report
