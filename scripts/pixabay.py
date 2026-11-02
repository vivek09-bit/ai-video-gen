#!/usr/bin/env python3
"""
Pixabay Stock Image API Integration for tube2dscript.

Provides functions and CLI commands to search and download royalty-free
stock images/photos from Pixabay for video scenes.
"""

import os
import re
import sys
import json
import urllib.parse
from pathlib import Path
from typing import List, Dict, Optional, Any

import requests
try:
    from dotenv import load_dotenv
    # Load .env file from project root if present
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    pass

PIXABAY_BASE_URL = "https://pixabay.com/api/"
DEFAULT_TIMEOUT = 15

# Common stop words to strip when simplifying queries that return 0 hits
STOP_WORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with",
    "by", "of", "from", "is", "was", "are", "were", "been", "being", "it",
    "its", "that", "this", "these", "those", "into", "over", "under", "about"
}


def get_api_key(explicit_key: Optional[str] = None) -> str:
    """Retrieve Pixabay API key from argument or environment."""
    key = explicit_key or os.getenv("PIXABAY_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "Pixabay API key not found. Please set PIXABAY_API_KEY in your .env file "
            "or pass it via --api-key. You can get a free API key at: "
            "https://pixabay.com/api/docs/"
        )
    return key


def clean_query_string(query: str, max_length: int = 100) -> str:
    """
    Pixabay limits query strings to 100 characters.
    Strip punctuation and truncate cleanly on word boundary.
    """
    # Replace non-alphanumeric (except spaces) with space
    cleaned = re.sub(r"[^\w\s]", " ", query).strip()
    # Normalize spaces
    cleaned = re.sub(r"\s+", " ", cleaned)
    if len(cleaned) > max_length:
        truncated = cleaned[:max_length]
        last_space = truncated.rfind(" ")
        if last_space > 0:
            cleaned = truncated[:last_space].strip()
        else:
            cleaned = truncated
    return cleaned


def simplify_query(query: str, max_words: int = 3) -> str:
    """
    Extract key content words from a descriptive prompt if exact search failed.
    """
    words = re.findall(r"\b\w+\b", query.lower())
    content_words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    if content_words:
        return " ".join(content_words[:max_words])
    return query[:50]


def get_best_image_url(hit: Dict[str, Any]) -> Optional[str]:
    """
    Extract the highest quality available image URL from a Pixabay hit.
    Prioritizes fullHDURL > largeImageURL > webformatURL.
    """
    return (
        hit.get("fullHDURL")
        or hit.get("largeImageURL")
        or hit.get("webformatURL")
        or hit.get("imageURL")
    )


def search_pixabay(
    query: str,
    api_key: Optional[str] = None,
    orientation: str = "horizontal",
    image_type: str = "photo",
    category: Optional[str] = None,
    safesearch: bool = True,
    per_page: int = 10,
    min_width: int = 1280,
    retry_with_simplified: bool = True,
) -> List[Dict[str, Any]]:
    """
    Search Pixabay for images matching the query.

    Args:
        query: Search keywords.
        api_key: Pixabay API key (optional if set in env).
        orientation: 'horizontal', 'vertical', or 'all'.
        image_type: 'photo', 'illustration', 'vector', or 'all'.
        category: Optional category filter (e.g., 'nature', 'backgrounds', 'places').
        safesearch: Enable SafeSearch filter.
        per_page: Number of results (3-200).
        min_width: Minimum width in pixels.
        retry_with_simplified: Whether to simplify query if 0 hits are returned.

    Returns:
        List of hit objects from Pixabay API.
    """
    key = get_api_key(api_key)
    cleaned_query = clean_query_string(query)

    params: Dict[str, Any] = {
        "key": key,
        "q": cleaned_query,
        "image_type": image_type,
        "orientation": orientation,
        "safesearch": "true" if safesearch else "false",
        "per_page": max(3, min(per_page, 200)),
        "order": "popular",
    }
    if min_width > 0:
        params["min_width"] = min_width
    if category:
        params["category"] = category

    response = requests.get(PIXABAY_BASE_URL, params=params, timeout=DEFAULT_TIMEOUT)
    if response.status_code != 200:
        raise RuntimeError(
            f"Pixabay API error ({response.status_code}): {response.text}"
        )

    data = response.json()
    hits = data.get("hits", [])

    # If 0 results and retry allowed, try simplified query (stripping filler words)
    if not hits and retry_with_simplified:
        simplified = simplify_query(cleaned_query)
        if simplified and simplified.lower() != cleaned_query.lower():
            params["q"] = simplified
            retry_res = requests.get(PIXABAY_BASE_URL, params=params, timeout=DEFAULT_TIMEOUT)
            if retry_res.status_code == 200:
                hits = retry_res.json().get("hits", [])

    return hits


def download_image(url: str, destination_path: str) -> str:
    """
    Download image from URL to local file path.
    """
    dest = Path(destination_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "tube2dscript-stock-fetcher/1.0"
    }

    with requests.get(url, headers=headers, stream=True, timeout=DEFAULT_TIMEOUT) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)

    return str(dest)


def fetch_and_save_image(
    query: str,
    output_path: str,
    api_key: Optional[str] = None,
    orientation: str = "horizontal",
    image_type: str = "photo",
    category: Optional[str] = None,
) -> Optional[str]:
    """
    Search Pixabay for the query and download the top image directly to output_path.

    Returns:
        The downloaded file path if successful, or None if no images were found.
    """
    hits = search_pixabay(
        query=query,
        api_key=api_key,
        orientation=orientation,
        image_type=image_type,
        category=category,
        per_page=5,
    )

    if not hits:
        print(f"[Pixabay] No images found for query: '{query}'")
        return None

    # Pick the first hit with a valid URL
    selected_url = None
    for hit in hits:
        url = get_best_image_url(hit)
        if url:
            selected_url = url
            break

    if not selected_url:
        print(f"[Pixabay] No valid image download URL in hits for query: '{query}'")
        return None

    saved_path = download_image(selected_url, output_path)
    print(f"[Pixabay] Saved image for '{query}' -> {saved_path}")
    return saved_path


def fetch_images_for_script(
    script_path: str,
    output_dir: str = "public/images",
    orientation: str = "horizontal",
    image_type: str = "photo",
    api_key: Optional[str] = None,
) -> Dict[str, str]:
    """
    Read scenes from script.json, search Pixabay for visual descriptions/queries,
    and download matching images into output_dir.

    Returns:
        Mapping of scene_id/index to local image file path.
    """
    with open(script_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenes = data.get("scenes", [])
    if not scenes and isinstance(data, list):
        scenes = data

    results: Dict[str, str] = {}
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    for i, scene in enumerate(scenes, start=1):
        scene_id = scene.get("id") or scene.get("scene_id") or f"scene-{i:03d}"
        
        # Priority for search query: visual_query > visual_prompt > visual > description > narration
        query = (
            scene.get("visual_query")
            or scene.get("visual_prompt")
            or scene.get("image_prompt")
            or scene.get("visual")
            or scene.get("description")
            or scene.get("narration")
            or scene.get("text")
            or ""
        )

        if not query:
            print(f"[Pixabay] Skipping scene {scene_id}: no text/query found.")
            continue

        filename = f"{scene_id}.jpg"
        target_path = str(out_dir_path / filename)

        try:
            saved = fetch_and_save_image(
                query=query,
                output_path=target_path,
                api_key=api_key,
                orientation=orientation,
                image_type=image_type,
            )
            if saved:
                results[str(scene_id)] = saved
        except Exception as e:
            print(f"[Pixabay] Error downloading image for {scene_id} ('{query}'): {e}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pixabay Stock Image Fetcher")
    parser.add_argument("--query", "-q", type=str, help="Search query keywords")
    parser.add_argument("--out", "-o", type=str, default="output.jpg", help="Output file path")
    parser.add_argument(
        "--orientation",
        choices=["horizontal", "vertical", "all"],
        default="horizontal",
        help="Image orientation (default: horizontal)",
    )
    parser.add_argument(
        "--type",
        choices=["photo", "illustration", "vector", "all"],
        default="photo",
        help="Image type (default: photo)",
    )
    parser.add_argument("--category", type=str, default=None, help="Pixabay category filter")
    parser.add_argument("--api-key", type=str, default=None, help="Explicit Pixabay API key")
    parser.add_argument("--script", type=str, default=None, help="Path to script.json for batch scene processing")
    parser.add_argument("--output-dir", type=str, default="public/images", help="Target folder for batch scene images")
    parser.add_argument("--test", action="store_true", help="Test Pixabay API connection")

    args = parser.parse_args()

    if args.test:
        print("[Pixabay] Testing API connection...")
        try:
            hits = search_pixabay("nature landscape", api_key=args.api_key, per_page=3)
            print(f"[Pixabay] Connection successful! Retrieved {len(hits)} test results.")
            if hits:
                print(f"Sample hit URL: {get_best_image_url(hits[0])}")
        except Exception as err:
            print(f"[Pixabay] Connection failed: {err}")
            sys.exit(1)
        return

    if args.script:
        print(f"[Pixabay] Processing batch script from {args.script}...")
        downloaded = fetch_images_for_script(
            script_path=args.script,
            output_dir=args.output_dir,
            orientation=args.orientation,
            image_type=args.type,
            api_key=args.api_key,
        )
        print(f"[Pixabay] Finished downloading {len(downloaded)} scenes.")
        return

    if not args.query:
        parser.print_help()
        sys.exit(1)

    fetch_and_save_image(
        query=args.query,
        output_path=args.out,
        api_key=args.api_key,
        orientation=args.orientation,
        image_type=args.type,
        category=args.category,
    )


if __name__ == "__main__":
    main()
