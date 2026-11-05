#!/usr/bin/env python3
"""Process, clean, and add natural Hindi punctuation to raw video scripts.

Raw scripts (e.g. from YouTube auto-subtitles or voice-to-text) often have:
1. Missing punctuation (no commas, full stops '।', or question marks '?').
2. Subtitle noise artifacts like '[संगीत]', '[चीखने की आवाज़]', '[Music]'.
3. Broken line wraps where every 4-8 words are split onto a new line.

This tool:
- Strips all subtitle noise and audio brackets.
- Reconstructs flowing paragraphs and natural sentence structures.
- Adds accurate Hindi punctuation (',', '।', '?', '!') using an LLM (DeepSeek / OpenAI)
  or an intelligent offline rule-based fallback.
- Can format directly into scene blocks ('###1|headline | VISUAL: tags') for video rendering.

Usage:
  # Clean and add punctuation to script.txt (prints to stdout):
  python scripts/process_script.py script.txt

  # Overwrite script.txt in place with the clean, punctuated version:
  python scripts/process_script.py script.txt --in-place

  # Save to a new file:
  python scripts/process_script.py script.txt -o script_clean.txt

  # Convert into video-ready scene blocks (for Remotion / new_video.py):
  python scripts/process_script.py script.txt --scenes --format reel
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# Pattern for audio/subtitle tags to remove
SUBTITLE_TAGS_PATTERN = re.compile(
    r"\[\s*(?:संगीत|हँसी|तालियाँ|चीख|चीखने की आवाज़|गहरी सांस|आवाज़|शोर|सांस|अचानक ज़ोर से सांस|music|applause|laughter|cheering|sound effects?|gasp|sigh)[^\]]*\]|"
    r"\(\s*(?:संगीत|हँसी|तालियाँ|चीख|चीखने की आवाज़|गहरी सांस|आवाज़|शोर|सांस|music|applause|laughter|cheering|sound effects?|gasp|sigh)[^\)]*\)",
    flags=re.IGNORECASE,
)

# Common Hindi sentence-ending verb tokens
HINDI_VERB_ENDINGS = {
    "था", "थी", "थे", "है", "हैं", "हूँ", "हूं", "होगा", "होगी", "होंगे",
    "गया", "गई", "गए", "गयी", "दिया", "दी", "दिए", "लिया", "ली", "लिए",
    "सका", "सकी", "सके", "पाया", "पाई", "पाए", "पड़ा", "पड़ी", "पड़े",
    "चाहिए", "रहा", "रही", "रहे", "चुका", "चुकी", "चुके",
}

# Clause starters that naturally need a preceding comma
HINDI_CLAUSE_STARTERS = {
    "लेकिन", "किंतु", "परंतु", "मगर", "क्योंकि", "ताकि", "जिसने", "जिसका",
    "जिसकी", "जिसके", "जिसमें", "जिससे", "जिन्होंने", "जिनका", "जिनकी",
    "जिनके", "जहां", "जहाँ", "जब", "जैसे", "तो",
}

# Question words that suggest a question mark '?' at the sentence end
HINDI_QUESTION_WORDS = {
    "क्यों", "कैसे", "क्या", "कब", "कहाँ", "कहा", "किसने", "किसको",
    "किसे", "किसलिए", "किसका", "किसकी", "किसके", "कौन",
}


def strip_subtitle_tags(text: str) -> str:
    """Remove audio cues, sound effect tags, and subtitle bracket noise."""
    cleaned = SUBTITLE_TAGS_PATTERN.sub(" ", text)
    # Also strip any isolated timestamp lines like '00:01:23,456'
    cleaned = re.sub(r"^\s*\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?\s*$", "", cleaned, flags=re.MULTILINE)
    return cleaned


def unwrap_transcript_lines(text: str) -> str:
    """Merge auto-caption line breaks into continuous paragraphs while preserving scene directives."""
    lines = text.splitlines()
    paragraphs: list[str] = []
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        # Keep empty lines as paragraph dividers
        if not stripped:
            if current_lines:
                paragraphs.append(" ".join(current_lines))
                current_lines = []
            continue

        # Keep directive lines intact (e.g. '###1|...', 'TITLE:', '@voice ...')
        if stripped.startswith("###") or stripped.upper().startswith("TITLE:") or stripped.startswith("@") or stripped.startswith("#"):
            if current_lines:
                paragraphs.append(" ".join(current_lines))
                current_lines = []
            paragraphs.append(stripped)
            continue

        current_lines.append(stripped)

    if current_lines:
        paragraphs.append(" ".join(current_lines))

    # Join paragraphs with clean double newlines
    result = "\n\n".join(p for p in paragraphs if p.strip())
    return result


def needs_punctuation(text: str) -> bool:
    """Check if Hindi text lacks adequate punctuation for speech synthesis."""
    words = text.split()
    if len(words) < 10:
        return False

    # Count Hindi / standard punctuation marks
    punct_count = len(re.findall(r"[,;!?।॥\.]", text))
    ratio = punct_count / max(1, len(words))

    # If fewer than 1 punctuation mark per 20 words, or zero question marks despite question words
    if ratio < 0.05:
        return True

    # Check for long unpunctuated stretches (> 25 consecutive words with no punctuation)
    chunks = re.split(r"[,;!?।॥\.]", text)
    max_unpunctuated = max((len(c.split()) for c in chunks), default=0)
    return max_unpunctuated > 25


def punctuate_hindi_rules(text: str) -> str:
    """Fast, robust offline heuristic punctuator for Hindi transcripts."""
    cleaned = strip_subtitle_tags(text)
    unwrapped = unwrap_transcript_lines(cleaned)

    paragraphs = unwrapped.split("\n\n")
    processed_paragraphs = []

    for para in paragraphs:
        # Don't touch directive lines
        if para.startswith("###") or para.upper().startswith("TITLE:") or para.startswith("@") or para.startswith("#"):
            processed_paragraphs.append(para)
            continue

        words = para.split()
        if not words:
            continue

        out_words: list[str] = []
        is_question_clause = False

        for i, w in enumerate(words):
            clean_w = w.strip(",।?!.:;")
            next_w = words[i + 1].strip(",।?!.:;") if i + 1 < len(words) else ""

            # Check if this word indicates a question
            if clean_w in HINDI_QUESTION_WORDS:
                is_question_clause = True

            # If next word is a clause starter, add comma after current word if missing
            if next_w in HINDI_CLAUSE_STARTERS and not w.endswith((",", "।", "?", "!", ".", ";")):
                out_words.append(w + ",")
                continue

            # If current word is a verb ending followed by a typical sentence starter or uppercase
            if clean_w in HINDI_VERB_ENDINGS and next_w in {
                "लेकिन", "और", "फिर", "अब", "तब", "जब", "वो", "वह", "यह", "ये",
                "मैं", "हम", "आप", "दोस्तों", "एक", "इस", "उस", "तभी", "तथा",
            } and not w.endswith((",", "।", "?", "!", ".")):
                mark = "?" if is_question_clause else "।"
                out_words.append(w + mark)
                is_question_clause = False
                continue

            out_words.append(w)

        para_text = " ".join(out_words)
        if not para_text.endswith(("।", "?", "!", ".")):
            para_text += "।" if re.search(r"[ऀ-ॿ]", para_text) else "."
        processed_paragraphs.append(para_text)

    return "\n\n".join(processed_paragraphs)


def punctuate_hindi_llm(
    text: str,
    writer_info: tuple[str, str, str | None, str] | None = None,
    chunk_size_words: int = 500,
) -> str:
    """Add punctuation and restore natural spoken pauses using an LLM."""
    try:
        import new_video
    except ImportError:
        new_video = None

    if writer_info is None and new_video is not None:
        writer_info = new_video.resolve_llm_writer()

    if not writer_info:
        # Fall back to offline rule-based punctuator
        return punctuate_hindi_rules(text)

    label, base_url, api_key, model = writer_info

    # 1. Clean subtitle tags and unwrap lines first
    cleaned = strip_subtitle_tags(text)
    unwrapped = unwrap_transcript_lines(cleaned)

    # 2. Divide into manageable narrative chunks (~400-600 words)
    paragraphs = unwrapped.split("\n\n")
    chunks: list[list[str]] = []
    current_chunk: list[str] = []
    current_words = 0

    for p in paragraphs:
        p_len = len(p.split())
        if current_chunk and (current_words + p_len > chunk_size_words):
            chunks.append(current_chunk)
            current_chunk = [p]
            current_words = p_len
        else:
            current_chunk.append(p)
            current_words += p_len

    if current_chunk:
        chunks.append(current_chunk)

    punctuated_chunks: list[str] = []
    total = len(chunks)

    for idx, chunk_paras in enumerate(chunks, 1):
        chunk_text = "\n\n".join(chunk_paras)
        # Skip directive blocks
        if chunk_text.startswith("###") or chunk_text.upper().startswith("TITLE:"):
            punctuated_chunks.append(chunk_text)
            continue

        prompt = f"""You are an expert Hindi literary and spoken script editor.
The following text is a raw transcript for a video narration. It contains missing punctuation,
erratic linebreaks, and missing pauses.

MANDATORY RULES:
1. Restore natural Hindi punctuation:
   - Commas (',') for natural breath groups, clauses, and pacing pauses.
   - Purna Viram (Devanagari full stop '।') at complete sentence conclusions.
   - Question marks ('?') for rhetorical or direct questions.
   - Exclamation marks ('!') for dramatic revelations.
2. Group related thoughts into natural, readable paragraphs separated by a blank line.
3. Remove any remaining subtitle noises like [संगीत], [चीख], [हँसी], etc.
4. STRICT FIDELITY: DO NOT change, remove, rewrite, summarize, or add ANY Hindi words.
   Every original word must be preserved exactly in its original order.
5. Return ONLY the clean, punctuated Hindi text. Do not add any English explanations or intros.

Raw Text:
{chunk_text}"""

        try:
            if total > 1:
                print(f"  [Punctuation] Processing section {idx}/{total} ({current_words} words)...", file=sys.stderr)
            res = new_video.ask_llm(
                base_url,
                api_key,
                model,
                prompt,
                temperature=0.1,
                max_tokens=4096,
            )
            punctuated_chunks.append(res.strip())
        except Exception as err:
            print(f"  [Warning] LLM punctuation failed on section {idx}: {err}. Using rule fallback.", file=sys.stderr)
            punctuated_chunks.append(punctuate_hindi_rules(chunk_text))

    return "\n\n".join(punctuated_chunks)


def format_into_scenes(
    text: str,
    target_scenes: int | None = None,
    words_per_scene: int = 25,
) -> str:
    """Convert punctuated Hindi text into standard video scene blocks (###N|headline | VISUAL: keywords)."""
    import story

    title, blocks = story.parse_blocks(text, scene_count=target_scenes or 999)
    if not blocks:
        return text

    out_lines: list[str] = []
    if title:
        out_lines.append(f"TITLE: {title}\n")

    for idx, b in enumerate(blocks, 1):
        headline = b.headline or f"Scene {idx}"
        visual = b.visual or "cinematic dramatic lighting, documentary shot"
        out_lines.append(f"###{idx}|{headline} | VISUAL: {visual}")
        out_lines.append(b.body)
        out_lines.append("")

    return "\n".join(out_lines).strip()


def process_script_file(
    file_path: Path,
    output_path: Path | None = None,
    in_place: bool = False,
    use_llm: bool = True,
    to_scenes: bool = False,
    target_scenes: int | None = None,
) -> str:
    """Process a script file, clean it, punctuate it, and optionally save it."""
    raw_text = file_path.read_text(encoding="utf-8")
    original_words = len(raw_text.split())

    print(f"[Script Processor] Reading {file_path.name} ({original_words} words)...", file=sys.stderr)

    # 1. Clean and punctuate
    if use_llm:
        punctuated = punctuate_hindi_llm(raw_text)
    else:
        punctuated = punctuate_hindi_rules(raw_text)

    # 2. Optionally format into scene blocks
    if to_scenes:
        punctuated = format_into_scenes(punctuated, target_scenes=target_scenes)

    # 3. Determine save target
    target_file = file_path if in_place else output_path
    if target_file:
        target_file.write_text(punctuated, encoding="utf-8")
        print(f"[Script Processor] Saved clean, punctuated script to: {target_file}", file=sys.stderr)

    return punctuated


def main():
    parser = argparse.ArgumentParser(
        description="Clean, unwrap, and add natural Hindi punctuation to raw video scripts."
    )
    parser.add_argument("script", help="Path to input script file (e.g. script.txt)")
    parser.add_argument("-o", "--out", dest="out", help="Output file path")
    parser.add_argument("-i", "--in-place", action="store_true", help="Overwrite the input script file in place")
    parser.add_argument("--no-llm", action="store_true", help="Force offline rule-based punctuation (skip LLM)")
    parser.add_argument("--scenes", action="store_true", help="Divide script into scene blocks (###1|headline | VISUAL: ...)")
    parser.add_argument("--target-scenes", type=int, default=None, help="Desired scene count when --scenes is passed")
    parser.add_argument("--preview", action="store_true", help="Print first 200 words preview to stdout")

    args = parser.parse_args()

    input_path = Path(args.script)
    if not input_path.exists():
        print(f"Error: {input_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out) if args.out else None
    result = process_script_file(
        input_path,
        output_path=out_path,
        in_place=args.in_place,
        use_llm=not args.no_llm,
        to_scenes=args.scenes,
        target_scenes=args.target_scenes,
    )

    if not args.in_place and not args.out:
        if args.preview:
            preview_words = result.split()[:200]
            print(" ".join(preview_words) + ("..." if len(result.split()) > 200 else ""))
        else:
            print(result)


if __name__ == "__main__":
    main()
