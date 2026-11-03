#!/usr/bin/env python3
"""Generate AI background images for script scenes.

Supports top providers in the $0.003 - $0.04 / image range:
  - fal.ai (FLUX 1.1 [pro] @ $0.04, FLUX.1 [dev] @ $0.025, FLUX.1 [schnell] @ $0.003)
  - Together AI (FLUX.1.1-pro @ $0.04, FLUX.1-dev @ $0.025, FLUX.1-schnell @ $0.003)
  - Replicate (flux-1.1-pro @ $0.04, recraft-v3 @ $0.04, flux-dev @ $0.025, flux-schnell @ $0.003)
  - OpenAI (dall-e-3 @ $0.04)

Environment variables (or .env file):
  FAL_KEY             e.g. key_id:secret or fal token
  TOGETHER_API_KEY    together.ai API key
  REPLICATE_API_TOKEN replicate API token
  OPENAI_API_KEY      openai API key
  IMAGE_PROVIDER      fal | together | replicate | openai (auto-detected by default)
  IMAGE_MODEL         specific model name or shortcut: flux-pro, flux-dev, flux-schnell, recraft
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import io
import os
import re
import sys
from pathlib import Path
from typing import Callable, Optional
import urllib.request
import urllib.error
import time

import requests
from PIL import Image


def load_env_file(path: str | Path = ".env") -> None:
    """Load key=value lines from a .env file if present."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = val


# Call at import time
load_env_file()

ANIMATION_STYLES: dict[str, dict[str, str]] = {
    "ink_doodle": {
        "name": "Ink Doodle & Whiteboard Sketch",
        "description": "Hand-drawn ink pen line art on textured parchment with cross-hatching and subtle watercolor washes (MinutePhysics, ASAPScience style)",
        "prefix": "Hand-drawn ink doodle sketch, intricate black ink pen line art on textured off-white paper, delicate cross-hatching shading, minimalist hand-crafted illustration: ",
        "suffix": ", subtle watercolor wash accents, clean negative space, artisanal storybook sketch, high quality 2D art.",
        "best_for": "educational, science, philosophy, trivia, explainer",
    },
    "flat_vector": {
        "name": "Flat 2D Vector Cartoon",
        "description": "Modern flat vector infographic illustration with bold geometric outlines and vibrant saturated color blocking (Kurzgesagt, Infographics Show style)",
        "prefix": "Modern flat 2D vector cartoon illustration, Kurzgesagt infographic aesthetic, crisp vector linework, bold geometric silhouettes: ",
        "suffix": ", vibrant saturated color blocking, smooth clean gradients, minimalist graphic design, high resolution 2D vector masterpiece.",
        "best_for": "science, tech, infographics, futuristic, comedy",
    },
    "documentary": {
        "name": "2D Documentary & Historical Illustration",
        "description": "Vintage archival etching, copperplate engraving texture, antique parchment and dramatic chiaroscuro lighting (Kings and Generals, OverSimplified style)",
        "prefix": "2D historical documentary illustration, vintage archival etching style, fine copperplate engraving texture, dramatic historic atmosphere: ",
        "suffix": ", antique sepia and rich muted tones, painterly digital art, dramatic chiaroscuro lighting, cinematic archival composition.",
        "best_for": "ancient_history, mythology, war, politics, deep dive",
    },
    "stickman": {
        "name": "Minimalist Stickman Animation",
        "description": "Expressive black line stick figure cartoon in viral webcomic style (Casually Explained, GradeAUnderA, Alan Becker, CGP Grey)",
        "prefix": "2D stickman animation style, viral comedy webcomic cartoon, bold black line stick figure illustration on clean white paper canvas: ",
        "suffix": ", Alan Becker Animator vs Animation and Casually Explained aesthetic, hilarious exaggerated stick poses, crisp stark black linework on pure white, high contrast 2D vector animation art, masterwork.",
        "best_for": "comedy, satire, life advice, casual explainers, storytelling",
    },
    "dark_cinematic": {
        "name": "Dark Cinematic 2D Illustration",
        "description": "Moody graphic novel illustration with deep ink shadows, thick fog, and intense chiaroscuro dread (Bedtime Stories, Wartime Stories style)",
        "prefix": "Cinematic dark graphic novel illustration, deep ink shadows, moody chiaroscuro lighting, thick eerie fog, intense contrast: ",
        "suffix": ", atmospheric 2D digital painting, Mike Mignola comic noir aesthetic, sharp focus, 8k resolution, haunting composition.",
        "best_for": "horror, mystery, true crime, supernatural thriller",
    },
    "anime_cel": {
        "name": "Anime Cel / Ghibli Aesthetic",
        "description": "Hand-painted scenic watercolor backgrounds, nostalgic lighting, and lush Studio Ghibli aesthetic",
        "prefix": "Nostalgic 2D anime cel illustration, Studio Ghibli and Makoto Shinkai aesthetic, hand-painted scenic watercolor background: ",
        "suffix": ", lush environmental details, soft golden hour lighting, billowing clouds, beautiful painterly anime scenery.",
        "best_for": "fantasy, drama, motivational, emotional stories",
    },
    "whiteboard": {
        "name": "Whiteboard Explainer Sketch",
        "description": "Clean dry-erase marker drawing on glossy white board with bold diagrams and doodle icons (RSA Animate style)",
        "prefix": "Clean whiteboard explainer illustration, dry erase marker sketch on glossy bright white board, clean outline diagram: ",
        "suffix": ", bold colored marker accents, educational presentation style, crisp clean linework, high contrast vector clarity.",
        "best_for": "business, finance, tutorials, study guides",
    },
    "vintage_retro": {
        "name": "1930s Vintage Rubber Hose",
        "description": "Classic 1930s rubber hose cartoon style with vintage film grain, halftone dots, and nostalgic character art (Cuphead style)",
        "prefix": "1930s vintage rubber hose cartoon illustration, classic retro animation aesthetic, vintage monochrome and muted Technicolor: ",
        "suffix": ", authentic film grain and scratches, halftone printing dot texture, whimsical nostalgic character design.",
        "best_for": "retro music, quirky trivia, vintage satire",
    },
}

STYLE_ALIASES: dict[str, str] = {
    "doodle": "ink_doodle",
    "ink": "ink_doodle",
    "ink_doodle": "ink_doodle",
    "sketch": "ink_doodle",
    "whiteboard_sketch": "ink_doodle",
    "vector": "flat_vector",
    "flat": "flat_vector",
    "cartoon": "flat_vector",
    "flat_vector": "flat_vector",
    "kurzgesagt": "flat_vector",
    "infographic": "flat_vector",
    "doc": "documentary",
    "history": "documentary",
    "historical": "documentary",
    "documentary": "documentary",
    "engraving": "documentary",
    "stick": "stickman",
    "stickman": "stickman",
    "stickfigure": "stickman",
    "stick_figure": "stickman",
    "stick_man": "stickman",
    "stick_animation": "stickman",
    "stickman_animation": "stickman",
    "stick_anim": "stickman",
    "stickanimation": "stickman",
    "dark": "dark_cinematic",
    "dark_cinematic": "dark_cinematic",
    "cinematic": "dark_cinematic",
    "horror": "dark_cinematic",
    "noir": "dark_cinematic",
    "anime": "anime_cel",
    "anime_cel": "anime_cel",
    "ghibli": "anime_cel",
    "cel": "anime_cel",
    "cel_shaded": "anime_cel",
    "marker": "whiteboard",
    "whiteboard": "whiteboard",
    "white_board": "whiteboard",
    "explainer": "whiteboard",
    "retro": "vintage_retro",
    "vintage": "vintage_retro",
    "vintage_retro": "vintage_retro",
    "rubberhose": "vintage_retro",
    "rubber_hose": "vintage_retro",
    "1930s": "vintage_retro",
}


def resolve_style_key(name: str | None) -> str | None:
    """Resolve an animation style name or alias to its canonical key."""
    if not name:
        return None
    k = name.strip().lower().replace("-", "_").replace(" ", "_")
    if k in ANIMATION_STYLES:
        return k
    return STYLE_ALIASES.get(k)


def get_animation_style(name: str | None) -> dict[str, str] | None:
    """Resolve an animation style name or alias to its style definition dict."""
    canonical = resolve_style_key(name)
    return ANIMATION_STYLES.get(canonical) if canonical else None


def is_stickman_style(name: str | None) -> bool:
    """Return True if the style name or alias maps to stickman animation."""
    return resolve_style_key(name) == "stickman"


def list_animation_styles() -> list[dict[str, str]]:
    """Return a list of all supported 2D animation styles with metadata."""
    result = []
    for key, data in ANIMATION_STYLES.items():
        result.append({
            "key": key,
            "name": data["name"],
            "description": data["description"],
            "best_for": data.get("best_for", ""),
        })
    return result


STYLE_PROMPT_PREFIX = (
    "Cinematic horror masterpiece, hyper-realistic eerie 2D digital painting, "
    "intense dark moody lighting, thick volumetric fog, deep unsettling shadows, "
    "ominous mystery, high contrast, terrifying atmosphere: "
)

STYLE_PROMPT_SUFFIX = (
    ", highly detailed background, cinematic composition, widescreen 16:9, "
    "dramatic lighting, sharp focus, 8k resolution, masterpiece."
)


# ---------------------------------------------------------------------------
# Theme -> prompt styling (Phase 5). The theme already drives the render palette;
# these fragments push the same colour/mood into the generated or retrieved visuals.
# ---------------------------------------------------------------------------
THEME_PROMPT_STYLE: dict[str, str] = {
    "blood_red": "crimson red color grading, blood-red glow, gothic horror palette",
    "amber_ember": "warm amber and ember tones, glowing firelight, orange haze",
    "cyber_neon": "cyberpunk neon palette, cyan and magenta glow, holographic light",
    "golden_radiance": "golden radiant glow, divine saffron light, sacred warmth",
    "amethyst_dream": "amethyst and violet palette, enchanted purple mist, dreamy glow",
    "emerald_forest": "emerald green palette, lush forest tones, jade light",
    "midnight_blue": "deep midnight blue palette, cold sapphire tones, moonlit ocean",
    "sunset_glow": "sunset rose and coral warmth, dusk glow, soft pink light",
    "monochrome_noir": "monochrome film-noir palette, high-contrast black and white",
    "toxic_lime": "toxic lime-green palette, acidic glow, radioactive haze",
    "solar_flare": "blazing solar flare palette, intense yellow-white light",
    "synthwave": "retro synthwave palette, neon pink and purple, 80s glow",
    "ice_glacier": "icy glacier palette, pale blue-white tones, frozen light",
    "royal_velvet": "royal velvet palette, regal purple and gold, opulent tones",
    "sepia_vintage": "sepia vintage palette, aged photograph tones, nostalgic grain",
    "pastel_whimsy": "soft pastel palette, gentle whimsical tones, airy light",
    "matrix_green": "matrix green palette, phosphor terminal glow, digital rain",
    "desert_dusk": "desert dusk palette, warm sand and bronze tones, twilight haze",
    "crimson_gothic": "crimson gothic palette, dark cathedral reds, ominous glow",
    "citrus_burst": "bright citrus palette, vivid orange and yellow, energetic light",
    "nordic_frost": "nordic frost palette, cool grey-blue tones, crisp icy light",
}

_THEME_SENTINELS = {"", "auto", "random", "none"}


def theme_style_fragment(theme: dict | str | None) -> str:
    """Return a short colour/mood fragment for a theme, or '' when unset/auto/random.

    Accepts either the theme dict stored in script.json or a bare theme key/alias.
    """
    if not theme:
        return ""
    if isinstance(theme, dict):
        key = str(theme.get("key") or theme.get("name") or "").strip().lower()
        accent = str(theme.get("accent") or "").strip()
    else:
        key = str(theme).strip().lower()
        accent = ""
    if key in _THEME_SENTINELS or key.startswith("#"):
        return ""
    fragment = THEME_PROMPT_STYLE.get(key)
    if fragment:
        return fragment
    # Unknown/custom theme: fall back to the human name and accent hex.
    human = key.replace("_", " ").strip()
    if accent:
        return f"{human} color grading, {accent} accent tones"
    return f"{human} color grading" if human else ""


def normalize_provider_name(name: str | None) -> str:
    """Normalize provider aliases to canonical names."""
    if not name:
        return ""
    n = name.lower().strip().replace("-", "").replace("_", "").replace(" ", "")
    if n in ("fetch", "stock", "free", "freeimage", "freeimages", "search", "unsplash", "lexica", "wiki"):
        return "fetch"
    if n in ("local", "lcm", "sd", "sd15", "localdiffusers", "sdcpp", "openvino", "onnx", "dreamshaper"):
        return "local"
    if n in ("diffusers", "diffuser", "sd", "stablediffusion", "sd15", "local"):
        return "local"
    if n in ("pollination", "pollinations", "pollinationsai", "freeai"):
        return "pollinations"
    if n in ("hf", "huggingface", "huggingfaceai", "gradio", "gradioclient", "hfspaces"):
        return "huggingface"
    if n in ("firework", "fireworks", "fireworksai"):
        return "fireworks"
    if n in ("together", "togetherai"):
        return "together"
    if n in ("fal", "falai"):
        return "fal"
    if n in ("replicate", "replicateai"):
        return "replicate"
    if n in ("openai", "dalle", "dalle3"):
        return "openai"
    if n in ("vast", "vastai", "vastgpu"):
        return "vast"
    return name.lower().strip()


HINDI_VISUAL_MAP: dict[str, str] = {
    # Fantasy, Magic & Fairytales
    "son pari": "fairy princess glowing magic enchanted wings",
    "sonpari": "fairy princess glowing magic enchanted wings",
    "pari": "fairy princess enchanted wings magic",
    "परी": "fairy princess enchanted glowing wings",
    "सोन परी": "fairy princess magical glowing wings sparkles",
    "सोनपरी": "fairy princess enchanted magical wings",
    "jadu": "magic spell glowing radiant illumination",
    "jadui": "magical enchanted glowing aura",
    "जादू": "magic glowing spell radiance",
    "जादुई": "magical enchanted glowing aura",
    "jadugar": "mystical sorcerer wizard casting spells",
    "जादूगर": "mystical sorcerer wizard casting spells",
    "titli": "glowing magical butterfly enchanted flowers",
    "तितली": "enchanted glowing butterfly flora",
    "khazana": "glowing treasure chest gold jewels radiant",
    "खजाना": "treasure chest glowing gold jewels",
    "mani": "glowing mystic crystal gemstone radiant",
    "मणि": "mystic crystal gemstone glowing aura",
    "heera": "sparkling diamond gemstone radiant",
    "हीरा": "sparkling diamond crystal",
    "taaj": "royal golden crown jewels",
    "ताज": "golden royal crown jewels",
    "singhasan": "majestic royal throne palace chamber",
    "सिंहासन": "majestic golden throne palace",
    "farishta": "luminous divine angel wings halo",
    "फ़रिश्ता": "luminous divine angel wings halo",
    "farishte": "luminous divine angels celestial",

    # Horror, Mystery & Supernatural
    "bhoot": "ghost spooky phantom apparition misty darkness",
    "भूत": "ghost eerie phantom spirit darkness",
    "chudail": "dark witch eerie sorceress shadows",
    "चुड़ैल": "witch eerie dark entity shadows",
    "aatma": "glowing spirit phantom apparition mist",
    "आत्मा": "spirit apparition glowing phantom",
    "jin": "mystical ancient genie spirit entity",
    "जिन्न": "ancient genie mystical phantom entity",
    "rakshas": "giant demon mythological dark monster beast",
    "राक्षस": "demon mythological dark monster beast",
    "danav": "giant mythical monster dark creature",
    "दानव": "mythical giant monster dark creature",
    "asur": "mythological demon warrior dark shadows",
    "असुर": "mythological demon warrior dark",
    "bhediya": "shadowy wolf howling full moon eerie",
    "भेड़िया": "shadowy wolf howling moon eerie",
    "samshan": "eerie graveyard cemetery tombstone fog night",
    "श्मशान": "eerie graveyard cemetery fog ruins",
    "kabristan": "ancient cemetery graveyard tombstone mist",
    "कब्रिस्तान": "cemetery graveyard tombstones fog",
    "tahkhana": "dark underground dungeon dungeon stone cellar",
    "तहखाना": "dark underground stone dungeon chamber",
    "zanjeer": "rusty iron chains shackles dark",
    "ज़ंजीर": "iron chains shackles dark stone",
    "dhokha": "dramatic confrontation shadows betrayal",
    "धोखा": "shadowy mystery dramatic confrontation",
    "train": "vintage steam locomotive train railway tracks eerie darkness",
    "station": "abandoned misty railway platform station vintage lamps",
    "ट्रेन": "vintage train railway tracks eerie misty night",
    "स्टेशन": "abandoned railway platform station vintage lamps misty",
    "kuldhara": "abandoned ruined cursed stone village of Kuldhara Rajasthan desert twilight",
    "कुलधरा": "abandoned ruined cursed stone village Kuldhara Rajasthan desert",
    "bhangarh": "cursed ancient Bhangarh fort ruins haunted ramparts dark eerie twilight",
    "भानगढ़": "cursed ancient Bhangarh fort ruins haunted ramparts dark twilight",
    "kankal": "ancient skeletal remains skull bones dark chamber",
    "कंकाल": "ancient human skeleton bones skull dark shadows",
    "khaufnak": "chilling horrifying eerie atmospheric darkness dread",
    "खौफनाक": "chilling horrifying eerie atmospheric shadows dread",
    "chhaya": "ominous dark shadow silhouette lurking misty corridor",
    "छाया": "ominous dark shadow silhouette lurking misty corridor",
    "saya": "eerie shadowy silhouette looming mysterious presence",
    "साया": "eerie shadowy phantom silhouette looming presence",
    "khooni": "blood-stained ominous cursed dark horror atmosphere",
    "खूनी": "blood-stained cursed dark horror ominous atmosphere",
    "shrap": "ancient mythical curse dark supernatural energy vortex",
    "श्राप": "ancient supernatural curse dark mystical vortex",
    "rahasya": "unsolved ancient mystery enigmatic relics glowing darkness",
    "रहस्य": "unsolved ancient mystery enigmatic relics glowing shadows",

    # Architecture, Structures & Interiors
    "mahal": "majestic royal palace castle architecture grand",
    "महल": "royal palace magical castle grand architecture",
    "haveli": "ancient heritage mansion dark manor architecture",
    "हवेली": "ancient heritage mansion architecture courtyard",
    "qila": "ancient stone fortress citadel castle ramparts",
    "किला": "ancient fortress stone citadel ramparts",
    "mandir": "ancient sacred monolithic temple stone carvings architecture",
    "मंदिर": "ancient sacred monolithic temple stone carvings architecture",
    "kailasa": "ancient monolithic Kailasa rock-cut temple carved from single cliff mountain",
    "कैलाश": "sacred Kailasa monolithic rock-cut temple carved stone architecture",
    "stambh": "ancient carved stone pillar monolithic column intricate relief",
    "स्तंभ": "ancient carved stone monolithic pillar column relief",
    "khambha": "monolithic carved stone pillar ancient architecture",
    "खंभा": "monolithic carved stone pillar ancient architecture",
    "shila": "massive monolithic granite stone rock block ancient carving",
    "शिला": "massive monolithic granite stone block ancient carving",
    "pathar": "ancient carved granite stone architecture archaeological relic",
    "पत्थर": "ancient carved granite stone architecture archaeological relic",
    "nakshi": "intricate ancient stone relief carvings sacred sacred geometry",
    "नक्काशी": "intricate ancient stone relief carvings sacred geometry",
    "shilpkar": "ancient master architect sculptor carving stone monument",
    "शिल्पकार": "ancient master sculptor architect carving stone monument",
    "ashram": "peaceful sacred sanctuary temple nature",
    "आश्रम": "serene sacred hermitage sanctuary nature",
    "darwaza": "mysterious ancient doorway carved gate entrance",
    "दरवाजा": "mysterious ancient door carved gate",
    "दरवाज़ा": "mysterious ancient door ornate gate",
    "khidki": "illuminated window glowing chamber evening",
    "खिड़की": "illuminated window glowing chamber",
    "kamra": "cozy antique room chamber interior",
    "कमरा": "atmospheric room chamber interior",
    "kaksh": "royal interior chamber hall",
    "कक्ष": "royal interior palace hall",
    "chhat": "rooftop skyline night moon view",
    "छत": "terrace rooftop cityscape view",
    "gali": "atmospheric narrow street vintage alleyway lantern",
    "गली": "atmospheric narrow vintage alley street",
    "sadak": "scenic winding road highway landscape",
    "सड़क": "scenic road highway pathway landscape",
    "rasta": "mysterious pathway trail winding woods",
    "रास्ता": "winding path trail through landscape",
    "pul": "ancient stone bridge crossing river",
    "पुल": "ancient stone arch bridge river",
    "bazaar": "bustling vibrant traditional marketplace bazaar",
    "बाज़ार": "bustling vibrant traditional marketplace",
    "बाजार": "colorful marketplace bazaar stalls",
    "dukan": "quaint storefront shop street scene",
    "दुकान": "quaint shop storefront street",
    "school": "vintage classroom academy library",
    "स्कूल": "vintage school classroom books",
    "hospital": "clinical medical interior illuminated",
    "अस्पताल": "clinical hospital interior chamber",
    "laboratory": "futuristic science laboratory glowing equipment",
    "प्रयोगशाला": "science laboratory glowing apparatus",

    # Nature, Landscapes & Wilderness
    "jangal": "enchanted dense forest woodland ancient trees canopy",
    "जंगल": "enchanted forest ancient trees misty canopy",
    "van": "lush green woodland forest landscape",
    "वन": "lush dense forest woodland nature",
    "gufa": "mysterious ancient cave cavern stone stalactites",
    "गुफा": "mysterious cavern cave interior glowing",
    "pahad": "majestic mountain peaks alpine scenery cliff",
    "पहाड़": "majestic mountain peaks landscape panorama",
    "parvat": "towering mountain range peaks snow sunrise",
    "पर्वत": "towering mountain peaks landscape",
    "himalaya": "sacred snowy Himalaya mountain peaks clouds",
    "हिमालय": "majestic snowy Himalaya peaks sacred",
    "samudra": "majestic ocean waves sea coastline horizon",
    "समुद्र": "majestic ocean sea coast horizon",
    "sagar": "deep blue ocean expanse waves horizon",
    "सागर": "deep blue ocean waters waves horizon",
    "nadi": "serene flowing river stream water landscape",
    "नदी": "serene river stream flowing landscape",
    "jharna": "majestic cascading waterfall river canyon",
    "झरना": "majestic cascading waterfall lush nature",
    "talab": "serene water pond reflections lotus nature",
    "तालाब": "serene pond water lake reflections",
    "jheel": "scenic mountain lake tranquil water landscape",
    "झील": "scenic tranquil lake water landscape",
    "dweep": "mysterious tropical island coast ocean",
    "द्वीप": "mysterious tropical island ocean coast",
    "ret": "golden desert sand dunes wind sweeping",
    "रेत": "golden desert sand dunes landscape",
    "marusthal": "vast desert wasteland dunes sunset",
    "मरुस्थल": "vast desert dunes sunset horizon",
    "gaon": "scenic rural village countryside rustic huts",
    "गाँव": "scenic rural village countryside scenery",
    "गांव": "rural village countryside rustic scenery",
    "shahar": "bustling city skyline metropolis skyscrapers lights",
    "शहर": "city skyline metropolis buildings night lights",

    # Anime, Action, Battle & Martial Arts
    "anime": "dynamic anime battle illustration energetic aura",
    "एनिमे": "dynamic anime illustration energy aura",
    "goku": "Goku Super Saiyan golden glowing energy aura charging battle",
    "गोकू": "Goku Super Saiyan golden aura energy battle",
    "naruto": "Naruto Uzumaki nine tails chakra fiery orange aura shinobi ninja",
    "नारूटो": "Naruto Uzumaki shinobi ninja orange chakra aura",
    "ninja": "shadowy shinobi ninja rooftop moonlight katana",
    "निंजा": "shinobi ninja rooftop stealth katana",
    "samurai": "noble samurai warrior katana armor cherry blossoms",
    "समुराई": "samurai warrior katana armor",
    "ladai": "explosive intense combat martial arts clash energy shockwave",
    "लड़ाई": "intense combat martial arts battle explosion",
    "yuddh": "epic war battlefield smoke fire warriors clash",
    "युद्ध": "epic war battlefield clash explosion smoke",
    "mukabla": "dramatic fierce warrior duel showdown arena",
    "मुकाबला": "fierce duel showdown warrior faceoff",
    "talwar": "gleaming sharp katana sword steel blade reflection",
    "तलवार": "gleaming sharp sword steel blade",
    "dhanush": "warrior aiming sacred golden bow arrow energy",
    "धनुष": "warrior aiming golden bow arrow",
    "teer": "energy arrow soaring glowing light trail",
    "तीर": "energy arrow glowing streak flight",
    "shakti": "radiant concentrated energy power blast aura",
    "शक्ति": "radiant power energy explosion aura",
    "hamla": "dynamic explosive combat attack strike",
    "हमला": "explosive combat attack strike",
    "prahar": "devastating martial arts impact strike shockwave",
    "प्रहार": "powerful impact strike shockwave",
    "dangal": "traditional wrestling arena dirt pit cheering crowd",
    "दंगल": "wrestling duel arena dirt pit",
    "pahalwan": "mighty muscular traditional wrestler champion",
    "पहलवान": "mighty wrestler champion",

    # Comedy, Humor & Lighthearted
    "mazaak": "funny comical character colorful playful mishap",
    "मज़ाक": "funny comical character cartoon mishap",
    "chutkula": "hilarious comedic scene smiling funny characters",
    "चुटकुला": "funny comedic scene cheerful laughing",
    "hansi": "cheerful joyful laughing people bright sunny day",
    "हँसी": "cheerful joyful laughing smiling faces",
    "gadbad": "chaotic hilarious kitchen mishap comical disaster",
    "गड़बड़": "chaotic comical kitchen mishap funny mess",
    "dhamaal": "wild energetic comical celebration party confetti",
    "धमाल": "energetic funny celebration party",

    # Education, Science & Discovery
    "vigyan": "fascinating scientific experiment glowing beaker laboratory",
    "विज्ञान": "scientific laboratory experiment research",
    "prayog": "science experiment laboratory microscope glassware",
    "प्रयोग": "science experiment laboratory research",
    "khoj": "historic archaeological exploration ancient map compass",
    "खोज": "exploration archaeological discovery ancient relics",
    "sooryamandal": "solar system planets orbiting glowing sun cosmos",
    "सौरमंडल": "solar system planets orbit glowing sun",
    "brahmand": "deep cosmos nebula infinite universe galaxy stars",
    "ब्रह्मांड": "deep cosmos nebula universe galaxy stars",
    "prithvi": "planet Earth from space blue marble glowing atmosphere",
    "पृथ्वी": "planet Earth from orbit space blue ocean",

    # Celestial, Cosmos & Sci-Fi
    "antriksh": "outer space cosmos galaxy nebula stars celestial",
    "अंतरिक्ष": "deep space nebula galaxy stars cosmos",
    "graha": "alien planet cosmos rings orbit celestial",
    "ग्रह": "alien planet deep space cosmos orbit",
    "robot": "futuristic cyborg android robot glowing circuits",
    "रोबोट": "futuristic android robot cyborg glowing",
    "alien": "extraterrestrial alien encounter glowing sci-fi",
    "एलियन": "extraterrestrial alien sci-fi encounter",
    "viman": "futuristic spacecraft ancient celestial flyer",
    "विमान": "futuristic spaceship celestial flyer",
    "scientist": "futuristic scientist laboratory research",
    "वैज्ञानिक": "futuristic scientist laboratory high-tech",

    # Mythology & Divine Figures
    "shiv": "Lord Shiva meditating sacred Kailash divine glowing aura",
    "शिव": "Lord Shiva meditating sacred divine Kailash aura",
    "mahadev": "Lord Mahadev divine sacred trident glowing",
    "महादेव": "Lord Mahadev divine trident sacred aura",
    "krishna": "Lord Krishna divine peacock feather flute Vrindavan radiant",
    "कृष्ण": "Lord Krishna divine flute peacock feather radiant",
    "ram": "Lord Rama divine warrior bow sacred radiance Ayodhya",
    "राम": "Lord Rama divine warrior sacred radiance",
    "hanuman": "Lord Hanuman divine strength sacred mountain power glowing",
    "हनुमान": "Lord Hanuman divine strength sacred radiance",
    "ganesha": "Lord Ganesha divine blessing sacred glowing",
    "गणेश": "Lord Ganesha divine sacred blessing aura",
    "durga": "Goddess Durga divine warrior lion sacred golden aura",
    "दुर्गा": "Goddess Durga divine warrior sacred radiance",
    "mahabharat": "ancient epic battle warrior chariot Kurukshetra",
    "महाभारत": "ancient epic battle warrior chariot Kurukshetra",
    "yoddha": "heroic ancient warrior gleaming armor sword shield",
    "योद्धा": "heroic warrior armor sword ancient champion",

    # Characters, Royalty & Common People
    "raja": "majestic king royal ornate robes palace court",
    "राजा": "majestic king royal ornate robes throne",
    "rani": "graceful queen royal crown ornate dress palace",
    "रानी": "graceful queen royal crown palace",
    "rajkumar": "noble prince royal castle kingdom",
    "राजकुमार": "noble prince royal castle kingdom",
    "rajkumari": "beautiful princess royal palace garden enchanted",
    "राजकुमारी": "beautiful princess royal enchanted garden",
    "kisan": "hardworking rural farmer scenic green crop fields",
    "किसान": "farmer working green countryside field",
    "bacha": "innocent child wonder wide eyes portrait",
    "बच्चा": "innocent child wonder happy",
    "chor": "shadowy stealth thief rogue dark alleyway",
    "चोर": "shadowy rogue thief cloaked dark",
    "sipahi": "ancient royal guard soldier armor spear",
    "सिपाही": "ancient royal guard soldier armor",
    "dost": "friends companionship camaraderie warm bond",
    "दोस्त": "friends together warm companionship bond",

    # Animals & Creatures
    "sher": "majestic lion predator wilderness savannah royal",
    "शेर": "majestic lion wilderness savannah royal",
    "baagh": "fierce royal Bengal tiger jungle foliage",
    "बाघ": "royal tiger jungle foliage stripes",
    "hathi": "majestic royal elephant decorated ancient",
    "हाथी": "majestic decorated elephant ancient temple",
    "ghoda": "noble steed galloping horse meadow landscape",
    "घोड़ा": "noble galloping horse countryside",
    "saap": "mystical serpent snake glowing eyes ancient",
    "सांप": "mystical serpent snake ancient ruins",
    "naag": "divine mystical cobra serpent sacred hood",
    "नाग": "divine cobra serpent mystical sacred aura",
    "nagin": "mystical serpent enchantress shapeshifter",
    "नागिन": "mystical serpent shapeshifter enchantress",
    "mor": "majestic peacock feathers radiant garden",
    "मोर": "majestic peacock vibrant feathers garden",
    "pakshi": "majestic soaring bird wings flight sky",
    "पक्षी": "majestic bird soaring open sky",

    # Weather, Atmosphere & Fire
    "aag": "roaring glowing fire flames sparks embers warm",
    "आग": "roaring fire flames embers glowing",
    "jwala": "intense radiant flame fire blaze",
    "ज्वाला": "radiant fiery blaze flames",
    "dhuwan": "atmospheric smoke haze rising embers",
    "धुआं": "atmospheric smoke haze mist",
    "barish": "heavy dramatic rain storm wet glistening reflections",
    "बारिश": "dramatic rain storm falling water reflections",
    "barsaat": "monsoon rain downpour puddles reflections",
    "बरसात": "monsoon rain dramatic downpour",
    "tufan": "dramatic tempest storm gale winds dark clouds",
    "तूफान": "dramatic storm tempest gale dark sky",
    "aandhi": "dust storm swirling winds dramatic sky",
    "आंधी": "dust storm swirling winds desert",
    "bijli": "dramatic lightning bolt strike dark stormy sky",
    "बिजली": "dramatic lightning strike stormy night",
    "kohra": "mysterious dense fog mist atmospheric woods",
    "कोहरा": "mysterious dense fog mist atmosphere",
    "dhund": "misty atmospheric haze morning landscape",
    "धुंध": "misty atmospheric haze soft light",
    "barf": "serene falling snow winter blizzard icy landscape",
    "बर्फ": "serene snow winter landscape frost",
    "hawa": "swirling wind breeze leaves swaying",
    "हवा": "swirling wind atmospheric breeze",
    "chand": "luminous glowing crescent moon starry night sky",
    "चांद": "luminous moon night sky stars",
    "chanda": "glowing moon night stars celestial",
    "चंदा": "glowing moon night stars",
    "chandrama": "full glowing moon night clouds",
    "चंद्रमा": "full luminous moon night sky",
    "suraj": "radiant golden sunrise morning sun rays horizon",
    "सूरज": "golden sunrise radiant sun rays horizon",
    "surya": "radiant golden sun dawn morning rays",
    "सूर्य": "radiant morning sun dawn golden rays",
    "tare": "sparkling stars constellations night sky cosmos",
    "तारे": "sparkling stars night sky cosmos",
    "sitare": "glowing starry night sky celestial constellations",
    "सितारे": "glowing starry night sky constellations",
    "aakash": "vast dramatic open sky clouds panorama",
    "आकाश": "vast open sky clouds panorama",
    "asman": "dramatic clouds open sky horizon",
    "आसमान": "dramatic sky clouds horizon",
    "badal": "dramatic billowing clouds atmospheric sky",
    "बादल": "billowing clouds atmospheric dramatic sky",

    # Objects & Relics
    "aaina": "mysterious ornate mirror reflection chamber",
    "आईना": "mysterious mirror reflection ornate frame",
    "darpan": "ancient reflective mirror mysterious glass",
    "दर्पण": "ancient reflective mirror ornate",
    "kanch": "shimmering glass crystal window shards",
    "कांच": "shimmering glass crystal reflections",
    "kitab": "ancient leatherbound grimoire book glowing glyphs",
    "किताब": "ancient book grimoire parchment pages",
    "pustak": "ancient scripture manuscript library",
    "पुस्तक": "ancient book manuscript library",
    "chirag": "antique brass oil lamp glowing flame warm",
    "चिराग": "antique oil lamp glowing flame",
    "diya": "traditional earthen clay lamp glowing flame diya",
    "दीया": "clay earthen lamp warm glowing flame",
    "mombatti": "flickering candle wax flame warm illumination",
    "मोमबत्ती": "flickering candle flame dark room",
    "roshni": "radiant luminous light beam illumination glowing",
    "रोशनी": "luminous radiant light beam glowing",
    "chamak": "sparkling glitter luminous glowing sheen",
    "चमक": "sparkling glitter luminous radiance",
    "prakash": "divine radiant light golden beam",
    "प्रकाश": "radiant light beam golden divine",
    "andhera": "deep shadows atmospheric dark contrast ominous",
    "अंधेरा": "deep shadows dark atmospheric contrast",
    "talwar": "gleaming ancient steel sword blade weapon",
    "तलवार": "gleaming steel sword blade ancient weapon",
    "dhanush": "ancient carved archer bow weapon",
    "धनुष": "ancient archer bow wooden weapon",
    "ghadi": "mysterious antique clock ticking time mechanism",
    "घड़ी": "mysterious antique clock gears time",
    "jahaz": "majestic wooden sailing ship high seas ocean",
    "जहाज": "majestic sailing ship ocean waters",
    "nao": "quaint wooden boat peaceful river water",
    "नाव": "quaint wooden boat serene river",

    # Narrative Beats & Themes
    "khwahish": "wistful desire longing dreamy stars ethereal",
    "ख्वाहिश": "dreamy longing wish ethereal stars",
    "pehchan": "mysterious identity discovery revelation mirror reflection",
    "पहचान": "mysterious identity discovery revelation",
    "kaid": "iron cage containment trapped bars imprisonment",
    "कैद": "iron cage bars containment trapped",
    "azadi": "soaring freedom open sky liberation wings",
    "आज़ादी": "freedom liberation soaring open sky",
    "khoj": "quest expedition discovery adventurous pathway",
    "खोज": "quest expedition exploration pathway",
    "rahasya": "cryptic ancient mystery enigmas shadows secrets",
    "रहस्य": "ancient cryptic mystery secrets shadows",
    "safar": "epic journey winding road horizon traveller",
    "सफर": "epic journey travel road horizon",
    "yatra": "pilgrimage expedition epic scenic journey",
    "यात्रा": "pilgrimage expedition scenic travel",
    "shuruat": "glorious dawn new beginning sunrise morning",
    "शुरुआत": "glorious dawn sunrise new beginning",
    "ant": "dramatic climax finale sunset resolution",
    "अंत": "dramatic climax finale resolution",
    "sapna": "ethereal surreal dreamscape fantasy wonderland",
    "सपना": "ethereal dreamscape surreal wonderland",
}


def extract_script_visual_keywords(
    scene: dict,
    topic: str = "",
    genre: str | None = None,
    script_keywords: list[str] | None = None,
) -> list[str]:
    """Extract script-specific visual entities prioritizing THIS SCENE's unique content.

    `script_keywords` are run-level research terms (Phase 1): they are added only as a light
    topical bias for scenes the writer gave no explicit visual keywords to, so a
    scene-specific description always wins.
    """
    try:
        import beats
        gp = beats.detect_genre(topic, genre or scene.get("genre"))
    except Exception:
        gp = None

    scene_keywords: list[str] = []
    stop_words = {
        "the", "and", "that", "this", "with", "from", "for", "are", "was", "were",
        "scene", "line", "part", "video", "clip", "text", "story", "write", "output",
        "step", "word", "words", "lead", "shots", "none", "true", "false", "have",
        "been", "what", "when", "where", "which", "will", "would", "could", "should",
    }

    # 0. PRIORITY 0: LLM-generated visual keywords (highest fidelity, scene-specific)
    llm_visual = scene.get("visualKeywords", "").strip()
    if llm_visual:
        # Translate any Hindi tokens in the LLM visual tags through the map
        llm_lower = llm_visual.lower()
        translated = []
        for hk, ev in HINDI_VISUAL_MAP.items():
            if hk in llm_lower:
                translated.append(ev)
        if translated:
            scene_keywords.extend(translated)
        else:
            # Use the raw English visual keywords directly
            scene_keywords.append(llm_visual)

    # 1. PRIORITY 1: Scene-specific text (headline, lines, story notes)
    scene_text_corpus = " ".join([
        scene.get("headline", ""),
        scene.get("_storyNote", ""),
        scene.get("_story", ""),
        " ".join(line.get("text", "") for line in scene.get("lines", [])),
    ]).lower()

    for hk, ev in HINDI_VISUAL_MAP.items():
        if hk in scene_text_corpus and ev not in scene_keywords:
            scene_keywords.append(ev)

    # Clean English words from scene-specific text
    raw_scene_ascii = re.findall(r"[a-zA-Z]{3,}", scene_text_corpus)
    scene_ascii = [w for w in raw_scene_ascii if w.lower() not in stop_words]
    if scene_ascii:
        scene_keywords.append(" ".join(scene_ascii[:5]))

    # 2. PRIORITY 2: Visual shot descriptors specific to this scene
    shots = scene.get("shots", [])
    for s in shots:
        v = s.get("visual", {})
        v_type = v.get("type", "")
        if v_type and v_type not in ("image", "black", "void"):
            if gp and v_type in gp.search_keywords_map:
                scene_keywords.append(gp.search_keywords_map[v_type])
            elif v_type == "house":
                scene_keywords.append("mansion house architecture")
            elif v_type == "forest":
                scene_keywords.append("forest woods trees")
            elif v_type == "corridor":
                scene_keywords.append("hallway corridor architecture")
            elif v_type == "candle":
                scene_keywords.append("glowing candle flame illumination")
            elif v_type == "grave":
                scene_keywords.append("ancient ruins graveyard tombstones")
            elif v_type == "figure":
                scene_keywords.append("mysterious silhouette figure character")
            elif v_type == "eyes":
                scene_keywords.append("glowing mysterious eyes darkness")
            elif v_type == "rain":
                scene_keywords.append("rain storm water reflections")
            elif v_type == "fog":
                scene_keywords.append("mysterious dense fog mist atmosphere")
            else:
                scene_keywords.append(v_type)

    # 3. PRIORITY 3: Topic-level keywords (as context modifier)
    topic_lower = (topic or "").lower()
    for hk, ev in HINDI_VISUAL_MAP.items():
        if hk in topic_lower and ev not in scene_keywords:
            scene_keywords.append(ev)

    raw_topic_ascii = re.findall(r"[a-zA-Z]{3,}", topic_lower)
    topic_ascii = [w for w in raw_topic_ascii if w.lower() not in stop_words]
    if topic_ascii:
        scene_keywords.append(" ".join(topic_ascii[:3]))

    # 4. PRIORITY 4: run-level research keywords. Only a light topical bias, and only for
    #    scenes the writer did not give explicit visual keywords to, so scene intent wins.
    if script_keywords and not llm_visual:
        for keyword in script_keywords[:5]:
            text = str(keyword).strip()
            if text and text not in scene_keywords:
                scene_keywords.append(text)

    # Default fallback if empty
    if not scene_keywords:
        if gp:
            scene_keywords.append(gp.default_keywords)
        else:
            scene_keywords.append("cinematic landscape atmosphere background")

    return scene_keywords


def build_search_queries(
    scene: dict,
    topic: str = "",
    genre: str | None = None,
    script_keywords: list[str] | None = None,
) -> list[str]:
    """Build a prioritized list of diverse search queries tailored to THIS scene."""
    try:
        import beats
        gp = beats.detect_genre(topic, genre or scene.get("genre"))
    except Exception:
        gp = None

    raw_keywords = extract_script_visual_keywords(scene, topic, genre, script_keywords)
    genre_name = gp.key if gp else "cinematic"
    beat = scene.get("beat", "")
    emotion = scene.get("emotion", "")

    # Query 1: Most specific scene visual query (entities translated from Hindi/script)
    q1_terms = list(dict.fromkeys(" ".join(raw_keywords).split()))
    q1 = " ".join(q1_terms[:6])

    # Query 2: Shot type + emotion + atmosphere
    shots = scene.get("shots", [])
    shot_types = [
        s.get("visual", {}).get("type", "")
        for s in shots
        if s.get("visual", {}).get("type") not in ("image", "black", "void")
    ]
    shot_str = " ".join(shot_types[:3])
    q2_elements = [item for item in [shot_str, emotion, beat, genre_name, "cinematic"] if item]
    q2 = " ".join(list(dict.fromkeys(" ".join(q2_elements).split()))[:6])

    # Query 3: Topic essence + genre default keywords
    clean_topic = re.sub(r"[^\x00-\x7F]+", " ", topic).strip()
    topic_first = " ".join(clean_topic.split()[:3]) if clean_topic else ""
    genre_defaults = gp.default_keywords if gp else "cinematic background landscape"
    q3 = f"{topic_first} {genre_defaults}".strip()
    q3 = " ".join(list(dict.fromkeys(q3.split()))[:6])

    queries = [q for q in [q1, q2, q3] if q and len(q) > 3]
    return list(dict.fromkeys(queries))


def make_search_keywords(
    scene: dict,
    topic: str = "",
    genre: str | None = None,
    script_keywords: list[str] | None = None,
) -> str:
    """Build clean, targeted English search keywords for stock/related image retrieval."""
    queries = build_search_queries(scene, topic, genre, script_keywords)
    return queries[0] if queries else "cinematic atmospheric background"


CHARACTER_TERMS = {
    "person", "people", "man", "woman", "child", "children", "boy", "girl",
    "figure", "silhouette", "character", "protagonist", "lead", "narrator",
    "traveler", "stranger", "wanderer", "villager", "hero", "warrior", "knight",
    "soldier", "king", "queen", "prince", "princess", "monarch", "emperor",
    "fairy", "sorcerer", "wizard", "mage", "witch", "sage", "monk", "priest",
    "ghost", "phantom", "spirit", "apparition", "demon", "monster", "beast",
    "creature", "dragon", "wolf", "tiger", "lion", "serpent", "snake",
    "entity", "detective", "inspector", "guard", "merchant", "farmer", "doctor",
    "scientist", "thief", "scholar", "specter", "vampire", "eyes", "cast",
}


def build_background_prompt(
    scene: dict,
    topic: str = "",
    genre: str | None = None,
    theme: dict | str | None = None,
    script_keywords: list[str] | None = None,
    animation_style: str | None = None,
) -> str:
    """Build the background/environment prompt component."""
    try:
        import beats
        gp = beats.detect_genre(topic, genre or scene.get("genre"))
    except Exception:
        gp = None

    beat = scene.get("beat", "").strip()
    emotion = scene.get("emotion", "").strip()

    # Extract scene-specific visual cues and filter out character words
    extracted_keywords = extract_script_visual_keywords(scene, topic, genre, script_keywords)
    bg_entities = [
        w for w in list(dict.fromkeys(" ".join(extracted_keywords).split()))
        if w.lower() not in CHARACTER_TERMS
    ][:7]
    bg_subject = " ".join(bg_entities) if bg_entities else "atmospheric environment"

    # Extract shot visual types for background
    visual_words = []
    shots = scene.get("shots", [])
    for s in shots:
        v = s.get("visual", {})
        v_type = v.get("type", "")
        if v_type and v_type not in ("image", "black", "void", "figure", "eyes", "cast"):
            visual_words.append(v_type)
            if v.get("lit"):
                visual_words.append("lit window warm illumination")
            if v.get("flicker"):
                visual_words.append("flickering flame light")
            if v.get("glow"):
                visual_words.append("glowing magical radiance")
            if v.get("lightning"):
                visual_words.append("lightning storm flashes")
            if v.get("fog"):
                visual_words.append("creeping mist and atmospheric fog")
            if v.get("moonlight"):
                visual_words.append("moonlight illumination")

    style_name = animation_style or scene.get("animationStyle") or scene.get("visualStyle")
    style_key = resolve_style_key(style_name)

    if style_key == "stickman":
        parts = [
            f"minimalist 2D stickman cartoon setting of {bg_subject}",
            "clean stark white paper canvas background",
            "minimal black pen line art outlines for environmental props",
            "spacious clean white negative space, zero 3D clutter, flat 2D line art",
        ]
        if visual_words:
            clean_v = [w for w in set(visual_words) if not any(k in w for k in ("mist", "fog", "creeping", "volumetric"))]
            if clean_v:
                parts.append(f"simple line art doodle of {', '.join(clean_v)}")
        return ", ".join(parts)

    if style_key == "documentary":
        parts = [
            f"2D historical archival etching of {bg_subject}",
            "aged antique parchment paper background with subtle vintage wear",
            "detailed copperplate engraving linework with fine cross-hatching",
            "classic 18th century historical print aesthetic, rich sepia and muted earth pigments",
        ]
        if visual_words:
            parts.append(f"featuring historical {', '.join(set(visual_words))}")
        return ", ".join(parts)

    if style_key == "flat_vector":
        parts = [
            f"modern flat 2D vector infographic environment depicting {bg_subject}",
            "crisp geometric vector silhouettes with clean razor-sharp outlines",
            "vibrant solid color blocking with subtle smooth gradients",
            "minimalist Kurzgesagt infographic aesthetic, zero photographic clutter",
        ]
        if visual_words:
            clean_v = [w for w in set(visual_words) if not any(k in w for k in ("mist", "fog", "creeping", "volumetric"))]
            if clean_v:
                parts.append(f"flat vector geometric {', '.join(clean_v)}")
        return ", ".join(parts)

    if style_key == "ink_doodle":
        parts = [
            f"hand-drawn ink pen sketch of {bg_subject} on textured warm ivory paper",
            "delicate black fountain pen line art with cross-hatching and stippled shading",
            "soft watercolor wash tint accents",
            "artisanal MinutePhysics notebook doodle aesthetic with clean negative space",
        ]
        if visual_words:
            parts.append(f"pen sketch of {', '.join(set(visual_words))}")
        return ", ".join(parts)

    if style_key == "dark_cinematic":
        parts = [
            f"dark graphic novel noir environment depicting {bg_subject}",
            "dramatic heavy black ink shadows with high-contrast chiaroscuro lighting",
            "moody atmospheric graphic composition, angular comic book silhouettes",
            "Mike Mignola comic noir aesthetic with deep ink blacks",
        ]
        if visual_words:
            parts.append(f"atmospheric {', '.join(set(visual_words))}")
        return ", ".join(parts)

    if style_key == "anime_cel":
        parts = [
            f"scenic 2D anime watercolor background depicting {bg_subject}",
            "hand-painted Studio Ghibli environmental landscape with lush painted details",
            "soft cinematic golden-hour atmospheric lighting and painterly sky",
            "classic Japanese animation scenery with delicate linework",
        ]
        if visual_words:
            parts.append(f"lush painted {', '.join(set(visual_words))}")
        return ", ".join(parts)

    if style_key == "whiteboard":
        parts = [
            f"clean whiteboard animation setting illustrating {bg_subject}",
            "glossy bright white dry-erase whiteboard background with subtle reflective sheen",
            "bold black dry-erase marker outline drawings and diagrammatic icons",
            "RSA Animate educational visual scribe aesthetic, high contrast clarity",
        ]
        if visual_words:
            parts.append(f"dry-erase marker doodle of {', '.join(set(visual_words))}")
        return ", ".join(parts)

    if style_key == "vintage_retro":
        parts = [
            f"classic 1930s rubber-hose cartoon background depicting {bg_subject}",
            "vintage hand-inked animation cel on aged painted cardboard",
            "authentic 1930s film grain, subtle dust scratches, and halftone dot printing texture",
            "Fleischer Studios and Cuphead retro theatrical cartoon backdrop, muted Technicolor",
        ]
        if visual_words:
            parts.append(f"vintage cartoon {', '.join(set(visual_words))}")
        return ", ".join(parts)

    parts = [bg_subject]
    if visual_words:
        unique_visuals = list(dict.fromkeys(visual_words))
        parts.append(f"featuring {', '.join(unique_visuals)}")
    if beat:
        parts.append(f"{beat} setting")
    if emotion:
        parts.append(f"{emotion} atmosphere")

    theme_fragment = theme_style_fragment(theme)
    if theme_fragment:
        parts.append(theme_fragment)

    genre_name = (genre or (gp.key if gp else "")).strip()
    if genre_name:
        parts.append(f"{genre_name} visual tone")

    return ", ".join(parts)


def build_character_prompt(
    scene: dict,
    topic: str = "",
    genre: str | None = None,
    characters_dict: dict | None = None,
    animation_style: str | None = None,
) -> str:
    """Build the character(s)/subject prompt component tailored to the visual animation style."""
    acting = scene.get("acting") or {}
    emotion = acting.get("emotion") or scene.get("emotion", "").strip() or "serious"
    head = acting.get("head")
    eyes_dir = acting.get("eyes")
    gesture = acting.get("gesture")

    style_name = animation_style or scene.get("animationStyle") or scene.get("visualStyle")
    style_key = resolve_style_key(style_name)

    # 1. Check if characters are explicitly cast in the scene
    cast = scene.get("cast", [])
    if cast:
        char_descriptions = []
        for member in cast:
            cast_id = member.get("id") or "lead"
            char_info = (characters_dict or {}).get(cast_id, {})
            name = char_info.get("name") or cast_id.title()
            hair = char_info.get("hairstyle")
            acc = char_info.get("accessory")

            member_acting = member.get("acting") or acting
            mem_emotion = member_acting.get("emotion") or emotion
            mem_head = member_acting.get("head") or head
            mem_eyes = member_acting.get("eyes") or eyes_dir
            mem_gesture = member_acting.get("gesture") or gesture

            pos_desc = "left side of frame" if member.get("x", 0.5) < 0.45 else ("right side of frame" if member.get("x", 0.5) > 0.55 else "center frame")
            gaze = mem_head or mem_eyes

            if style_key == "stickman":
                stick_features = []
                if hair and hair not in ("bald", "none"):
                    stick_features.append(f"simple doodle {hair} hair lines")
                if acc and acc != "none":
                    stick_features.append(f"minimalist stick {acc}")
                feat_str = f" ({', '.join(stick_features)})" if stick_features else ""
                desc = (
                    f"minimalist 2D black line stick figure of {name}{feat_str}, round circle head, "
                    f"expressive {mem_emotion} cartoon face, thin stick torso and limbs in exaggerated {mem_gesture or 'animated'} pose, "
                    f"positioned at {pos_desc}"
                )
                if gaze:
                    desc += f", head turned {gaze}"

            elif style_key == "documentary":
                hair_desc = f"{hair} hair" if hair else "period-appropriate hairstyle"
                acc_desc = f", wearing {acc}" if acc and acc != "none" else ""
                desc = (
                    f"authentic 2D historical archival figure of {name}, {hair_desc}{acc_desc}, "
                    f"solemn {mem_emotion} expression, period attire, fine copperplate engraved detail, "
                    f"positioned at {pos_desc}"
                )
                if gaze:
                    desc += f", looking {gaze}"
                if mem_gesture and mem_gesture != "none":
                    desc += f", {mem_gesture} posture"

            elif style_key == "flat_vector":
                hair_desc = f"{hair} hair" if hair else "geometric hairstyle"
                acc_desc = f", stylized {acc}" if acc and acc != "none" else ""
                desc = (
                    f"clean modern 2D flat vector character of {name}, {hair_desc}{acc_desc}, "
                    f"crisp geometric silhouette, bold solid color fills, stylized {mem_emotion} face, "
                    f"{mem_gesture or 'dynamic'} pose, positioned at {pos_desc}"
                )
                if gaze:
                    desc += f", facing {gaze}"

            elif style_key == "ink_doodle":
                hair_desc = f"{hair} hair" if hair else "hand-drawn hair"
                acc_desc = f", with {acc}" if acc and acc != "none" else ""
                desc = (
                    f"hand-drawn black ink pen sketch of {name}, intricate cross-hatching, {hair_desc}{acc_desc}, "
                    f"lively {mem_emotion} facial expression, organic textured ink linework, "
                    f"positioned at {pos_desc}"
                )
                if gaze:
                    desc += f", looking {gaze}"
                if mem_gesture and mem_gesture != "none":
                    desc += f", {mem_gesture} gesture"

            elif style_key == "dark_cinematic":
                hair_desc = f"{hair} hair" if hair else "shadowed hair"
                acc_desc = f", with {acc}" if acc and acc != "none" else ""
                desc = (
                    f"dramatic graphic novel noir character of {name}, {hair_desc}{acc_desc}, "
                    f"heavy black ink shadows, sharp rim lighting, intense {mem_emotion} expression, "
                    f"positioned at {pos_desc}"
                )
                if gaze:
                    desc += f", glancing {gaze}"
                if mem_gesture and mem_gesture != "none":
                    desc += f", {mem_gesture} posture"

            elif style_key == "anime_cel":
                hair_desc = f"{hair} hair" if hair else "anime hairstyle"
                acc_desc = f", anime {acc}" if acc and acc != "none" else ""
                desc = (
                    f"vibrant 2D anime cel animation character of {name}, stylized {hair_desc}{acc_desc}, "
                    f"large expressive anime eyes with emotional highlights, {mem_emotion} expression, "
                    f"clean anime linework, positioned at {pos_desc}"
                )
                if gaze:
                    desc += f", head turned {gaze}"
                if mem_gesture and mem_gesture != "none":
                    desc += f", {mem_gesture} action"

            elif style_key == "whiteboard":
                hair_desc = f"{hair} hair" if hair else "marker-sketched hair"
                acc_desc = f", drawn with {acc}" if acc and acc != "none" else ""
                desc = (
                    f"dry-erase marker whiteboard cartoon of {name}, bold marker linework, {hair_desc}{acc_desc}, "
                    f"expressive simple {mem_emotion} cartoon face, educational marker sketch pose, "
                    f"positioned at {pos_desc}"
                )
                if gaze:
                    desc += f", facing {gaze}"
                if mem_gesture and mem_gesture != "none":
                    desc += f", gesturing {mem_gesture}"

            elif style_key == "vintage_retro":
                hair_desc = f"{hair} hair" if hair else "classic 1930s cartoon hair"
                acc_desc = f", wearing {acc}" if acc and acc != "none" else ""
                desc = (
                    f"classic 1930s rubber-hose cartoon character of {name}, vintage pie-eyes, white gloves, {hair_desc}{acc_desc}, "
                    f"bouncy noodle limbs in exaggerated {mem_emotion} pose, retro animation cel aesthetic, "
                    f"positioned at {pos_desc}"
                )
                if gaze:
                    desc += f", looking {gaze}"

            else:
                hair_desc = f"{hair} hair" if hair else "detailed hairstyle"
                acc_desc = f", wearing {acc}" if acc and acc != "none" else ""
                desc = f"{name}, {hair_desc}{acc_desc}, {mem_emotion} expression, positioned at {pos_desc}"
                if gaze:
                    desc += f", looking {gaze}"
                if mem_gesture and mem_gesture != "none":
                    desc += f", {mem_gesture} gesture"

            char_descriptions.append(desc)
        return "; ".join(char_descriptions)

    # 2. Check for figure or eyes visual shot types
    shots = scene.get("shots", [])
    shot_types = [s.get("visual", {}).get("type") for s in shots if s.get("visual")]
    if "figure" in shot_types:
        if style_key == "stickman":
            return f"mysterious 2D black stick figure silhouette in distance, {emotion} posture, minimalist line art composition"
        if style_key == "documentary":
            return f"distant 2D archival historical figure silhouette, antique copperplate engraving scale, {emotion} presence"
        if style_key == "flat_vector":
            return f"clean geometric 2D flat vector human silhouette in distance, crisp solid color shape, minimal graphic design"
        if style_key == "ink_doodle":
            return f"delicate hand-inked pen silhouette figure in distance, fine cross-hatching, organic textured paper sketch"
        if style_key == "dark_cinematic":
            return f"dramatic graphic novel noir silhouette shrouded in heavy ink shadows, sharp high-contrast rim lighting, {emotion} presence"
        if style_key == "anime_cel":
            return f"scenic 2D anime character silhouette in distance against dramatic painted sky, cel-shaded anime aesthetic"
        if style_key == "whiteboard":
            return f"quick dry-erase marker cartoon silhouette sketched in background, bold marker outline on clean whiteboard"
        if style_key == "vintage_retro":
            return f"1930s vintage rubber-hose cartoon silhouette in distance, retro animation cel proportions, vintage film texture"
        return f"mysterious shadowy silhouette figure in distance, {emotion} presence, cinematic ambient lighting"

    if "eyes" in shot_types:
        if style_key == "stickman":
            return f"giant comically exaggerated cartoon eyes popping out, {emotion} shock, hilarious stick figure reaction"
        if style_key == "documentary":
            return f"intense historical engraved portrait focus on human eyes, antique copperplate cross-hatching, deep {emotion} gaze"
        if style_key == "flat_vector":
            return f"stylized flat 2D vector eye graphic close-up, sharp geometric lines, clean solid colors, expressive {emotion} glance"
        if style_key == "ink_doodle":
            return f"intricate hand-drawn fountain pen eye sketch, fine black ink cross-hatching, expressive {emotion} gaze"
        if style_key == "dark_cinematic":
            return f"piercing eyes emerging from deep black comic noir shadows, stark chiaroscuro lighting, intense {emotion} stare"
        if style_key == "anime_cel":
            return f"detailed 2D anime eye close-up, vibrant multi-layered iris highlights, emotional {emotion} expression, beautiful anime art"
        if style_key == "whiteboard":
            return f"expressive dry-erase marker cartoon eyes sketch, bold black marker strokes with simple {emotion} highlights"
        if style_key == "vintage_retro":
            return f"classic 1930s vintage cartoon pie-cut eyes, dramatic wide cartoon expression, retro rubber-hose animation detail"
        return f"intense focused eyes in dramatic lighting, {emotion} expressive gaze"

    # 3. Check for character entities in scene narrative/corpus
    extracted_keywords = extract_script_visual_keywords(scene, topic, genre)
    found_chars = [
        w for w in list(dict.fromkeys(" ".join(extracted_keywords).split()))
        if w.lower() in CHARACTER_TERMS
    ]
    if found_chars:
        char_label = " ".join(found_chars[:3])
        if style_key == "stickman":
            return f"hilarious 2D black line stick figure character ({char_label}), round head with exaggerated {emotion} facial expression, funny stick limbs in dynamic {gesture or 'animated'} gesture"
        if style_key == "documentary":
            return f"archival 2D historical illustration of {char_label}, antique copperplate engraving realism, period attire, {emotion} presence"
        if style_key == "flat_vector":
            return f"modern flat 2D vector character illustration depicting {char_label}, crisp geometric shapes, solid color fills, {emotion} posture"
        if style_key == "ink_doodle":
            return f"hand-drawn black ink pen sketch of {char_label}, fine cross-hatched shading, expressive {emotion} character demeanor"
        if style_key == "dark_cinematic":
            return f"moody graphic novel character illustration of {char_label}, dramatic heavy ink shadows, intense {emotion} demeanor"
        if style_key == "anime_cel":
            return f"expressive 2D anime character illustration of {char_label}, clean cel linework, stylized hair and vibrant {emotion} expression"
        if style_key == "whiteboard":
            return f"dry-erase marker whiteboard illustration of {char_label}, bold marker outlines and diagrammatic cartoon charm"
        if style_key == "vintage_retro":
            return f"vintage 1930s rubber-hose cartoon character of {char_label}, classic retro animation styling, whimsical {emotion} pose"
        return f"{char_label} character, {emotion} expression and dramatic posture"

    # 4. Solitary/Uninhabited scene fallback
    if style_key == "stickman":
        return f"expressive 2D black line stick figure narrator, round circle head with {emotion} cartoon expression, thin stick arms and legs in funny animated posture, engaging the audience, center frame"
    if style_key == "documentary":
        return "uninhabited archival historical setting, antique copperplate engraving detail, stillness and historic gravity, no people present"
    if style_key == "flat_vector":
        return "uninhabited flat vector graphic composition, clean geometric architecture and negative space, modern infographic clarity, no people present"
    if style_key == "ink_doodle":
        return "uninhabited hand-inked architectural sketch, delicate fountain pen lines and cross-hatching, serene paper stillness, no people present"
    if style_key == "dark_cinematic":
        return "uninhabited graphic novel noir scenery, deep atmospheric ink shadows, silent moody tension, no people present"
    if style_key == "anime_cel":
        return "serene scenic 2D anime background art, beautiful hand-painted landscape, peaceful Ghibli atmosphere, no people present"
    if style_key == "whiteboard":
        return "clean whiteboard diagram composition, educational marker charts and visual concepts, bright organized layout, no people present"
    if style_key == "vintage_retro":
        return "uninhabited 1930s vintage cartoon background scenery, antique painted animation cel backdrop, nostalgic retro stillness, no people present"
    return "uninhabited scenic composition, no people present, solitary perspective, quiet stillness"


def make_prompt(
    scene: dict,
    topic: str = "",
    genre: str | None = None,
    visual_style: str | None = None,
    index: int = 0,
    script_keywords: list[str] | None = None,
    theme: dict | str | None = None,
    animation_style: str | None = None,
    characters_dict: dict | None = None,
) -> str:
    """Construct an evocative visual prompt structured with background first and characters second."""
    try:
        import beats
        gp = beats.detect_genre(topic, genre or scene.get("genre"))
    except Exception:
        gp = None

    # Resolve animation style profile if selected
    anim_style = (
        get_animation_style(animation_style)
        or get_animation_style(visual_style)
        or get_animation_style(scene.get("animationStyle"))
    )
    resolved_style = animation_style or visual_style or scene.get("animationStyle")

    bg_prompt = build_background_prompt(
        scene,
        topic=topic,
        genre=genre,
        theme=theme,
        script_keywords=script_keywords,
        animation_style=resolved_style,
    )
    char_prompt = build_character_prompt(
        scene,
        topic=topic,
        genre=genre,
        characters_dict=characters_dict,
        animation_style=resolved_style,
    )

    # Persist prompt components on scene for downstream inspection and tooling
    scene["backgroundPrompt"] = bg_prompt
    scene["characterPrompt"] = char_prompt

    if anim_style:
        prefix = anim_style["prefix"]
        suffix = anim_style["suffix"]
    else:
        prefix = visual_style or (gp.visual_style_prefix if gp else STYLE_PROMPT_PREFIX)
        suffix = gp.visual_style_suffix if gp else STYLE_PROMPT_SUFFIX

    suffix_clean = suffix if (suffix.startswith(",") or suffix.startswith(" ")) else f", {suffix}"
    full_prompt = f"{prefix}Background: {bg_prompt.rstrip('. ')}; Characters: {char_prompt.rstrip('. ')}{suffix_clean}"
    scene["prompt"] = full_prompt
    return full_prompt


def detect_provider(requested_provider: str | None = None) -> tuple[str, str, str]:
    """Detect available image provider, key, and default model.

    Returns (provider_name, api_key, default_model).
    Defaults to local Diffusers generation ('local', 'segmind/SSD-1B').
    """
    raw_env_provider = os.environ.get("IMAGE_PROVIDER", "").strip()
    provider = normalize_provider_name(requested_provider or raw_env_provider)
    model = os.environ.get("IMAGE_MODEL", "").strip()

    if provider in ("fetch", "free", "stock", "unsplash", "lexica", "wiki"):
        return "fetch", "free", model or "free-search"
    if provider == "local":
        local_model = model if (model and model.lower() not in ("flux", "default", "free-search", "none")) else "segmind/SSD-1B"
        return "local", "local", local_model
    if provider == "pollinations":
        key = os.environ.get("POLLINATIONS_API_KEY", "")
        return "pollinations", key or "free", model or "default"
    if provider in ("huggingface", "hf", "gradio"):
        key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY", "")
        return "huggingface", key, model or "black-forest-labs/FLUX.1-schnell"
    if provider == "fireworks":
        key = os.environ.get("FIREWORKS_API_KEY") or os.environ.get("FIREWORK_API_KEY", "")
        return "fireworks", key, model or "flux-1-schnell-fp8"
    if provider == "together":
        key = os.environ.get("TOGETHER_API_KEY") or os.environ.get("TOGETHERAI_API_KEY", "")
        return "together", key, model or "black-forest-labs/FLUX.1-schnell"
    if provider == "fal":
        key = os.environ.get("FAL_KEY") or os.environ.get("FAL_API_KEY", "")
        return "fal", key, model or "fal-ai/flux/schnell"
    if provider == "replicate":
        key = os.environ.get("REPLICATE_API_TOKEN") or os.environ.get("REPLICATE_API_KEY", "")
        return "replicate", key, model or "black-forest-labs/flux-schnell"
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        return "openai", key, model or "dall-e-3"
    if provider == "vast":
        key = os.environ.get("VAST_API_KEY", "")
        return "vast", key or "vast", model or "black-forest-labs/flux-3-image-klein"

    # Default to local SSD-1B generation
    return "local", "local", model or "segmind/SSD-1B"


def save_image_data(result: str | bytes, dest: Path) -> None:
    """Save image to disk from bytes, direct URL, data URI, or base64 string."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(result, bytes):
        dest.write_bytes(result)
        return

    result_str = result.strip()
    if result_str.startswith("http://") or result_str.startswith("https://"):
        req = urllib.request.Request(result_str, headers={"User-Agent": "tube2dscript/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            dest.write_bytes(resp.read())
    elif result_str.startswith("data:"):
        # data:image/png;base64,...
        _, b64_data = result_str.split(",", 1)
        dest.write_bytes(base64.b64decode(b64_data))
    else:
        # Base64 string payload
        dest.write_bytes(base64.b64decode(result_str))


def fetch_free_related_image(
    scene: dict,
    topic: str = "",
    index: int = 0,
    genre: str | None = None,
    script_keywords: list[str] | None = None,
) -> bytes:
    """Fetch matching high-res related image for free via Lexica / Unsplash / Wikimedia / Openverse with Pollinations fallback."""
    queries = build_search_queries(scene, topic, genre=genre, script_keywords=script_keywords)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 tube2dscript/1.0",
        "Accept": "application/json,image/*,*/*",
    }

    # Generate a deterministic unique seed for this scene so images are diverse and distinct
    seed_str = f"{topic}_{scene.get('id', '')}_{scene.get('headline', '')}_{index}_{time.time()}"
    unique_seed = int(hashlib.md5(seed_str.encode("utf-8")).hexdigest()[:8], 16)

    # Iterate through prioritized search queries
    for query in queries:
        encoded_query = urllib.parse.quote(query)

        # 1. Try Lexica AI Image Database (High quality 2D artistic visuals indexed by semantic match)
        try:
            lexica_url = f"https://lexica.art/api/v1/search?q={encoded_query}"
            req = urllib.request.Request(lexica_url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                images = data.get("images", [])
                if images:
                    selected = images[index % len(images)]
                    img_url = selected.get("src") or selected.get("srcSmall")
                    if img_url:
                        img_req = urllib.request.Request(img_url, headers=headers, method="GET")
                        with urllib.request.urlopen(img_req, timeout=25) as img_resp:
                            raw = img_resp.read()
                            if raw.startswith(b"\x89PNG") or raw.startswith(b"\xff\xd8\xff") or raw.startswith(b"RIFF"):
                                return raw
        except Exception:
            pass

        # 2. Try Unsplash search API (High aesthetic photography & landscapes)
        try:
            unsplash_url = f"https://unsplash.com/napi/search/photos?query={encoded_query}&per_page=15&orientation=landscape"
            req = urllib.request.Request(unsplash_url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                results = data.get("results", [])
                if results:
                    selected = results[index % len(results)]
                    img_url = (
                        selected.get("urls", {}).get("regular")
                        or selected.get("urls", {}).get("full")
                        or selected.get("urls", {}).get("small")
                    )
                    if img_url:
                        img_req = urllib.request.Request(img_url, headers=headers, method="GET")
                        with urllib.request.urlopen(img_req, timeout=25) as img_resp:
                            raw = img_resp.read()
                            if raw.startswith(b"\x89PNG") or raw.startswith(b"\xff\xd8\xff") or raw.startswith(b"RIFF"):
                                return raw
        except Exception:
            pass

        # 3. Try Wikimedia Commons search API
        try:
            wiki_url = f"https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch={encoded_query}&gsrnamespace=6&prop=imageinfo&iiprop=url&format=json&gsrlimit=12"
            req = urllib.request.Request(wiki_url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                pages = list(data.get("query", {}).get("pages", {}).values())
                if pages:
                    selected_page = pages[index % len(pages)]
                    infos = selected_page.get("imageinfo", [])
                    if infos and "url" in infos[0]:
                        img_req = urllib.request.Request(infos[0]["url"], headers=headers, method="GET")
                        with urllib.request.urlopen(img_req, timeout=25) as img_resp:
                            raw = img_resp.read()
                            if raw.startswith(b"\x89PNG") or raw.startswith(b"\xff\xd8\xff") or raw.startswith(b"RIFF"):
                                return raw
        except Exception:
            pass

        # 4. Try Openverse Public Image API
        try:
            openverse_url = f"https://api.openverse.org/v1/images/?q={encoded_query}&license_type=all&page_size=10"
            req = urllib.request.Request(openverse_url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                results = data.get("results", [])
                if results:
                    selected = results[index % len(results)]
                    img_url = selected.get("url")
                    if img_url:
                        img_req = urllib.request.Request(img_url, headers=headers, method="GET")
                        with urllib.request.urlopen(img_req, timeout=25) as img_resp:
                            raw = img_resp.read()
                            if raw.startswith(b"\x89PNG") or raw.startswith(b"\xff\xd8\xff") or raw.startswith(b"RIFF"):
                                return raw
        except Exception:
            pass

    # 5. Free generation via Pollinations AI with clean prompt and unique seed
    try:
        prompt = make_prompt(scene, topic, genre=genre, index=index)
        return generate_pollinations(prompt, model="default", seed=unique_seed)
    except Exception:
        pass

    # 6. Guaranteed Fallback: High-res seeded CDN image
    try:
        cdn_seed = (unique_seed % 9999) + 1
        cdn_url = f"https://picsum.photos/seed/{cdn_seed}/1280/720"
        req = urllib.request.Request(cdn_url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
            if raw.startswith(b"\x89PNG") or raw.startswith(b"\xff\xd8\xff") or raw.startswith(b"RIFF"):
                return raw
    except Exception:
        pass

    raise RuntimeError(f"Could not retrieve or generate image for scene {index + 1}")


_LOCAL_DIFFUSION_PIPELINE = None
_LOCAL_DIFFUSION_MODEL_NAME = None


def is_turbo_model(model_name: str | None) -> bool:
    """Check if model name refers to SDXL-Turbo."""
    if not model_name:
        return False
    m = model_name.lower().strip()
    return "turbo" in m or "sdxl-turbo" in m


def is_ssd_model(model_name: str | None) -> bool:
    """Check if model name refers to Segmind SSD-1B."""
    if not model_name:
        return False
    m = model_name.lower().strip()
    return "ssd" in m or "ssd-1b" in m or "segmind" in m


def is_flux_model(model_name: str | None) -> bool:
    """Check if model name refers to FLUX model."""
    if not model_name:
        return False
    m = model_name.lower().strip()
    return "flux" in m or "klein" in m


def is_klein_model(model_name: str | None) -> bool:
    """Check if model name specifically refers to FLUX.2 Klein."""
    if not model_name:
        return False
    m = model_name.lower().strip()
    return "klein" in m or "flux-2" in m or "flux2" in m


def get_local_diffusion_pipeline(model_name: str = ""):
    """Lazily load and cache local diffusion pipeline (SD 1.5 LCM, SSD-1B + LCM-LoRA, SDXL-Turbo) on CPU."""
    global _LOCAL_DIFFUSION_PIPELINE, _LOCAL_DIFFUSION_MODEL_NAME
    target = (model_name or "").strip()
    if _LOCAL_DIFFUSION_PIPELINE is not None and _LOCAL_DIFFUSION_MODEL_NAME == target:
        return _LOCAL_DIFFUSION_PIPELINE

    if _LOCAL_DIFFUSION_PIPELINE is not None:
        cleanup_local_pipeline()

    import torch
    from diffusers import (
        AutoPipelineForText2Image,
        DiffusionPipeline,
        LCMScheduler,
        StableDiffusionXLPipeline,
    )

    target = (model_name or "").strip()

    # 1. Handle SDXL-Turbo (1-step adversarial distilled SDXL)
    if is_turbo_model(target):
        turbo_id = "stabilityai/sdxl-turbo"
        print(f"  [Local SDXL-Turbo] Loading 1-step model '{turbo_id}' on CPU...")
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(
                turbo_id,
                torch_dtype=torch.float32,
                safety_checker=None,
            )
            _LOCAL_DIFFUSION_PIPELINE = pipe
        except Exception as e:
            print(f"  [Local SDXL-Turbo Notice] Failed to load SDXL-Turbo ({e}). Falling back to LCM DreamShaper...")
            target = "SimianLuo/LCM_Dreamshaper_v7"

    # 2. Handle Segmind SSD-1B with LCM-LoRA (pruned SDXL model, 50% smaller)
    elif is_ssd_model(target):
        ssd_id = "segmind/SSD-1B"
        print(f"  [Local SSD-1B] Loading fast 1B model '{ssd_id}' with LCM-LoRA on CPU...")
        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                ssd_id,
                torch_dtype=torch.float32,
            )
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            try:
                pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl", weight_name="pytorch_lora_weights.safetensors")
            except Exception:
                pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
            _LOCAL_DIFFUSION_PIPELINE = pipe
        except Exception as e:
            print(f"  [Local SSD-1B Notice] Failed to load SSD-1B ({e}). Falling back to LCM DreamShaper...")
            target = "SimianLuo/LCM_Dreamshaper_v7"

    # 3. Handle FLUX.2 Klein 4B Q4_0 / FLUX if specified
    elif is_flux_model(target):
        is_klein = is_klein_model(target) or "q4" in target.lower()
        print(f"  [Local FLUX] Initializing FLUX pipeline for '{target}'...")
        pipe = None
        if is_klein:
            try:
                from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel, GGUFQuantizationConfig
                repo_id = "unsloth/FLUX.2-klein-4B-GGUF"
                filename = "flux-2-klein-4b-Q4_0.gguf"
                print(f"  [Local FLUX.2 Klein Q4_0] Loading quantized transformer ({repo_id}/{filename})...")
                transformer = Flux2Transformer2DModel.from_single_file(
                    f"https://huggingface.co/{repo_id}/blob/main/{filename}",
                    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float32),
                    torch_dtype=torch.float32,
                    config="black-forest-labs/FLUX.2-klein-4B",
                    subfolder="transformer",
                )
                pipe = Flux2KleinPipeline.from_pretrained(
                    "black-forest-labs/FLUX.2-klein-4B",
                    transformer=transformer,
                    torch_dtype=torch.float32,
                )
                _LOCAL_DIFFUSION_PIPELINE = pipe
            except ImportError:
                print(
                    "  [Local FLUX.2 Klein Notice] Native Diffusers Flux2KleinPipeline requires diffusers >= 0.37.0.\n"
                    "  For maximum local speed on CPU/iGPU, run stable-diffusion.cpp with SD_LOCAL_URL=http://localhost:8080\n"
                    "  or cloud GPU via --vast.\n"
                    "  Falling back to optimized LCM DreamShaper..."
                )
                target = "SimianLuo/LCM_Dreamshaper_v7"
            except Exception as e:
                print(f"  [Local FLUX.2 Klein Notice] GGUF Klein load notice ({e}). Falling back to LCM DreamShaper...")
                target = "SimianLuo/LCM_Dreamshaper_v7"

        if pipe is None and _LOCAL_DIFFUSION_PIPELINE is None and target != "SimianLuo/LCM_Dreamshaper_v7":
            try:
                from diffusers import FluxPipeline
                base_model_id = "black-forest-labs/FLUX.1-schnell" if "schnell" in target.lower() else target
                pipe = FluxPipeline.from_pretrained(
                    base_model_id,
                    torch_dtype=torch.float32,
                )
                _LOCAL_DIFFUSION_PIPELINE = pipe
            except Exception as e:
                print(f"  [Local FLUX Notice] FLUX load notice ({e}). Falling back to optimized LCM / SD pipeline...")
                target = "SimianLuo/LCM_Dreamshaper_v7"

    # 3. Default / SD 1.5 LCM Pipeline
    if _LOCAL_DIFFUSION_PIPELINE is None:
        if not target or target in ("lcm", "lcm-sd15", "default", "sd15", "local", "lcm-dreamshaper"):
            target = "SimianLuo/LCM_Dreamshaper_v7"
        elif target in ("lcm-lora", "dreamshaper-lcm", "dreamshaper"):
            target = "Lykon/dreamshaper-7"

        print(f"  [Local LCM] Loading pipeline '{target}' on CPU (cached for subsequent scenes)...")
        pipe = None
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(
                target,
                torch_dtype=torch.float32,
                safety_checker=None,
            )
        except Exception:
            try:
                pipe = AutoPipelineForText2Image.from_pretrained(
                    target,
                    torch_dtype=torch.float32,
                    safety_checker=None,
                    local_files_only=True,
                )
            except Exception:
                try:
                    pipe = DiffusionPipeline.from_pretrained(
                        target,
                        custom_pipeline="latent_consistency_txt2img",
                        custom_revision="main",
                        torch_dtype=torch.float32,
                        safety_checker=None,
                    )
                except Exception:
                    pipe = DiffusionPipeline.from_pretrained(
                        target,
                        torch_dtype=torch.float32,
                        safety_checker=None,
                        local_files_only=True,
                    )

        if target == "Lykon/dreamshaper-7" or "lcm-lora" in target.lower():
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
            pipe.fuse_lora()
        elif hasattr(pipe, "scheduler") and not isinstance(pipe.scheduler, LCMScheduler):
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        _LOCAL_DIFFUSION_PIPELINE = pipe

    _LOCAL_DIFFUSION_MODEL_NAME = target
    _LOCAL_DIFFUSION_PIPELINE.to("cpu")
    if hasattr(_LOCAL_DIFFUSION_PIPELINE, "enable_attention_slicing"):
        _LOCAL_DIFFUSION_PIPELINE.enable_attention_slicing()

    return _LOCAL_DIFFUSION_PIPELINE


def get_local_lcm_pipeline(model_name: str = ""):
    """Alias for backwards compatibility."""
    return get_local_diffusion_pipeline(model_name)


def cleanup_local_pipeline() -> None:
    """Release local diffusion pipeline and free CPU/GPU RAM."""
    global _LOCAL_DIFFUSION_PIPELINE, _LOCAL_DIFFUSION_MODEL_NAME
    if _LOCAL_DIFFUSION_PIPELINE is not None:
        try:
            import gc
            import torch
            del _LOCAL_DIFFUSION_PIPELINE
            _LOCAL_DIFFUSION_PIPELINE = None
            _LOCAL_DIFFUSION_MODEL_NAME = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            _LOCAL_DIFFUSION_PIPELINE = None
            _LOCAL_DIFFUSION_MODEL_NAME = None


def calculate_sd_dimensions(width: int, height: int, is_flux: bool = False) -> tuple[int, int]:
    """Calculate multiple-of-8 SD latent dimensions for an arbitrary target aspect ratio."""
    aspect = width / max(1, height)
    if is_flux:
        # Flux Transformer runs O(N^2) in sequence length; use fast lightweight latents for CPU
        if aspect <= 0.6:  # 9:16 portrait / reel
            return 384, 640
        elif aspect <= 0.85:  # 4:5 portrait
            return 384, 480
        elif aspect <= 1.15:  # 1:1 square
            return 448, 448
        elif aspect <= 1.45:  # 4:3 classic
            return 512, 384
        elif aspect <= 1.95:  # 16:9 widescreen
            return 640, 384
        else:  # 21:9 ultrawide / cinema
            return 704, 320

    if aspect <= 0.6:  # 9:16 portrait / reel
        return 432, 768
    elif aspect <= 0.85:  # 4:5 portrait
        return 464, 576
    elif aspect <= 1.15:  # 1:1 square
        return 512, 512
    elif aspect <= 1.45:  # 4:3 classic
        return 640, 480
    elif aspect <= 1.95:  # 16:9 widescreen
        return 768, 432
    else:  # 21:9 ultrawide / cinema
        return 896, 384


def generate_local_lcm(
    prompt: str,
    model: str = "",
    seed: int | None = None,
    steps: int = 4,
    guidance_scale: float = 1.5,
    width: int = 1280,
    height: int = 720,
    sd_width: int | None = None,
    sd_height: int | None = None,
) -> bytes:
    """Generate image locally via SD 1.5 + LCM (Diffusers / CPU) or local SD API."""
    import io
    from PIL import Image

    # Check if a local HTTP server endpoint is configured (e.g. sd.cpp or SD WebUI / Forge / A1111)
    local_url = os.environ.get("SD_LOCAL_URL") or os.environ.get("SD_CPP_URL") or os.environ.get("SD_WEBUI_URL")
    if local_url:
        local_url = local_url.rstrip("/")
        endpoints = [
            f"{local_url}/v1/images/generations",
            f"{local_url}/sdapi/v1/txt2img",
            f"{local_url}/txt2img",
        ]
        payload = json.dumps({
            "prompt": prompt,
            "steps": steps,
            "width": width,
            "height": height,
            "cfg_scale": guidance_scale,
            "seed": seed if seed is not None else -1,
            "response_format": "b64_json",
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        for ep in endpoints:
            try:
                req = urllib.request.Request(ep, data=payload, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=120) as resp:
                    res_data = json.loads(resp.read().decode("utf-8"))
                    if "data" in res_data and isinstance(res_data["data"], list) and res_data["data"]:
                        first = res_data["data"][0]
                        if "b64_json" in first:
                            return base64.b64decode(first["b64_json"])
                        if "url" in first:
                            u_req = urllib.request.Request(first["url"], headers={"User-Agent": "tube2dscript/1.0"})
                            with urllib.request.urlopen(u_req, timeout=60) as u_resp:
                                return u_resp.read()
                    if "images" in res_data and res_data["images"]:
                        return base64.b64decode(res_data["images"][0])
                    if "image" in res_data:
                        return base64.b64decode(res_data["image"])
            except Exception:
                continue

    # In-process CPU pipeline using diffusers
    import torch

    is_flux = is_flux_model(model)
    is_turbo = is_turbo_model(model)
    pipe = get_local_diffusion_pipeline(model)
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    # Calculate dimensions
    if sd_width and sd_height:
        sd_w, sd_h = sd_width, sd_height
    else:
        sd_w, sd_h = calculate_sd_dimensions(width, height, is_flux=is_flux)

    # Distilled FLUX.2 Klein 4B / FLUX requires ~4 steps and guidance_scale=1.0
    if is_turbo:
        inf_steps = 1
        cfg = 0.0
    elif is_flux:
        inf_steps = steps if (steps and steps > 0) else 4
        cfg = guidance_scale if guidance_scale is not None else 1.0
    else:
        inf_steps = steps if (steps and steps > 0) else 4
        cfg = guidance_scale if guidance_scale else 1.5

    try:
        if is_flux:
            result = pipe(
                prompt=prompt,
                width=sd_w,
                height=sd_h,
                num_inference_steps=inf_steps,
                guidance_scale=cfg,
                generator=generator,
                max_sequence_length=128,
            )
        elif is_turbo:
            result = pipe(
                prompt=prompt,
                width=sd_w,
                height=sd_h,
                num_inference_steps=inf_steps,
                guidance_scale=cfg,
                generator=generator,
            )
        else:
            result = pipe(
                prompt=prompt,
                width=sd_w,
                height=sd_h,
                num_inference_steps=inf_steps,
                guidance_scale=cfg,
                generator=generator,
            )
    except Exception as e:
        print(f"  [Local Generation Warning] {e}. Falling back to default generation...")
        result = pipe(
            prompt=prompt,
            width=sd_w,
            height=sd_h,
            num_inference_steps=1 if is_turbo else 4,
            guidance_scale=0.0 if is_turbo else 1.5,
            generator=generator,
        )

    img = result.images[0]
    if (img.width, img.height) != (width, height):
        img = img.resize((width, height), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_pollinations(
    prompt: str,
    model: str = "default",
    api_key: str = "",
    seed: int | None = None,
    width: int = 1280,
    height: int = 720,
) -> bytes:
    """Generate free AI image via Pollinations without requiring a paid key."""
    safe_prompt = prompt.strip().replace("\n", " ")
    encoded_prompt = urllib.parse.quote(safe_prompt)
    seed_val = seed if seed is not None else int(time.time() * 1000) % 1000000

    # If model is explicitly given and not "flux"/"default", we pass it; otherwise default free model
    model_param = ""
    if model and model.lower() not in ("default", "flux", "free"):
        model_param = f"&model={urllib.parse.quote(model)}"

    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}{model_param}&nologo=true&seed={seed_val}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) tube2dscript/1.0",
        "Accept": "image/webp,image/png,image/jpeg,*/*",
    }
    if api_key and api_key != "free":
        headers["Authorization"] = f"Bearer {api_key}"

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            if attempt > 0:
                time.sleep(attempt * 2)
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=120) as resp:
                content_type = resp.headers.get("Content-Type", "")
                raw = resp.read()
                if (
                    raw.startswith(b"\x89PNG")
                    or raw.startswith(b"\xff\xd8\xff")
                    or raw.startswith(b"RIFF")
                    or "image" in content_type
                ):
                    return raw
                raise RuntimeError(
                    f"Pollinations returned non-image response: {raw[:100]}"
                )
        except urllib.error.HTTPError as err:
            last_err = err
            if err.code == 402:
                # If 402, retry without any model param
                url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=576&nologo=true&seed={seed_val}"
            time.sleep(2.0)
        except Exception as err:
            last_err = err
            time.sleep(1.5)

    raise last_err or RuntimeError("Failed to generate image via Pollinations")


def generate_gradio_hf(
    prompt: str,
    token: str = "",
    model: str = "black-forest-labs/FLUX.1-schnell",
    width: int = 1024,
    height: int = 576,
) -> bytes:
    """Generate image via Hugging Face Spaces using Gradio Client with multi-space failover."""
    from gradio_client import Client

    # HF Spaces strictly enforce max dimensions of 1024x1024
    hf_w = width
    hf_h = height
    if hf_w > 1024 or hf_h > 1024:
        if width > height:
            hf_w, hf_h = 1024, 576
        elif height > width:
            hf_w, hf_h = 576, 1024
        else:
            hf_w, hf_h = 1024, 1024

    spaces = [
        (
            "black-forest-labs/FLUX.1-schnell",
            "/infer",
            {
                "prompt": prompt,
                "seed": 0,
                "randomize_seed": True,
                "width": hf_w,
                "height": hf_h,
                "num_inference_steps": 4,
            },
        ),
        (
            "stabilityai/stable-diffusion-3.5-large-turbo",
            "/infer",
            {
                "prompt": prompt,
                "negative_prompt": "blurry, distorted, low quality, cartoon, artifacts",
                "seed": 0,
                "randomize_seed": True,
                "width": hf_w,
                "height": hf_h,
                "guidance_scale": 0.0,
                "num_inference_steps": 4,
            },
        ),
    ]
    if model and any(k in model.lower() for k in ("sd3", "turbo", "stability")):
        spaces.reverse()

    last_error = None
    clean_token = token.strip() if token and token != "free" and not token.startswith("...") else None
    for space_id, endpoint, kwargs in spaces:
        try:
            client = Client(space_id, token=clean_token)
            result = client.predict(**kwargs, api_name=endpoint)
            img_val = result[0] if isinstance(result, (tuple, list)) else result
            if isinstance(img_val, dict):
                img_val = img_val.get("path") or img_val.get("url")
            if img_val and os.path.isfile(str(img_val)):
                return Path(img_val).read_bytes()
            if isinstance(img_val, str) and (img_val.startswith("http://") or img_val.startswith("https://")):
                req = urllib.request.Request(img_val, headers={"User-Agent": "tube2dscript/1.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return resp.read()
        except Exception as e:
            last_error = e
            print(f"  [Gradio HF Notice] Space '{space_id}' failed ({e}). Trying next space...")
            continue

    raise RuntimeError(f"All Hugging Face Gradio Spaces failed. Last error: {last_error}")


def generate_huggingface(prompt: str, key: str, model: str = "", width: int = 1024, height: int = 576) -> bytes:
    """Generate free image via Hugging Face Gradio Client / Spaces."""
    return generate_gradio_hf(prompt, token=key, model=model, width=width, height=height)


def generate_fireworks(prompt: str, key: str, model: str) -> str | bytes:
    """Generate image via Fireworks AI."""
    model_map = {
        "flux-pro": "accounts/fireworks/models/flux-1-dev-fp8",
        "flux-dev": "accounts/fireworks/models/flux-1-dev-fp8",
        "flux-schnell": "accounts/fireworks/models/flux-1-schnell-fp8",
    }
    target_model = model_map.get(model, model) or "flux-1-schnell-fp8"

    workflow_url = "https://api.fireworks.ai/inference/v1/workflows/accounts/fireworks/models/flux-1-schnell-fp8/text_to_image"
    payload = json.dumps(
        {
            "prompt": prompt,
            "aspect_ratio": "16:9",
            "output_format": "png",
        }
    ).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    data: dict | None = None
    try:
        req = urllib.request.Request(workflow_url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
            if raw.startswith(b"\x89PNG") or raw.startswith(b"\xff\xd8\xff"):
                return raw
            data = json.loads(raw.decode("utf-8"))
    except Exception:
        gen_url = "https://api.fireworks.ai/inference/v1/image_generations"
        gen_payload = json.dumps(
            {
                "model": target_model,
                "prompt": prompt,
                "aspect_ratio": "16:9",
            }
        ).encode("utf-8")
        req = urllib.request.Request(gen_url, data=gen_payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
            if raw.startswith(b"\x89PNG") or raw.startswith(b"\xff\xd8\xff"):
                return raw
            data = json.loads(raw.decode("utf-8"))

    if isinstance(data, dict):
        if "image" in data:
            if isinstance(data["image"], dict) and "url" in data["image"]:
                return data["image"]["url"]
            if isinstance(data["image"], str):
                return data["image"]
        if "data" in data and data["data"]:
            first = data["data"][0]
            if "url" in first and first["url"]:
                return first["url"]
            if "b64_json" in first and first["b64_json"]:
                return first["b64_json"]
        if "choices" in data and data["choices"]:
            first = data["choices"][0]
            if "url" in first:
                return first["url"]
            if "image" in first:
                return first["image"]
        if "samples" in data and data["samples"]:
            first = data["samples"][0]
            if isinstance(first, str):
                return first
            if "url" in first:
                return first["url"]
    raise RuntimeError(f"Fireworks response did not contain image data: {data}")


def generate_fal(prompt: str, key: str, model: str) -> str:
    """Generate image via fal.ai and return image URL."""
    # Resolve shorthand model names
    model_map = {
        "flux-pro": "fal-ai/flux-pro/v1.1",
        "flux-dev": "fal-ai/flux/dev",
        "flux-schnell": "fal-ai/flux/schnell",
    }
    target_model = model_map.get(model, model) or "fal-ai/flux-pro/v1.1"

    url = f"https://fal.run/{target_model}"
    payload = json.dumps(
        {
            "prompt": prompt,
            "image_size": "landscape_16_9",
            "num_images": 1,
            "enable_safety_checker": True,
        }
    ).encode("utf-8")

    headers = {
        "Authorization": f"Key {key}",
        "Content-Type": "application/json",
    }

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    images = data.get("images", [])
    if images and "url" in images[0]:
        return images[0]["url"]
    if "image" in data and "url" in data["image"]:
        return data["image"]["url"]
    raise RuntimeError(f"fal.ai response did not contain image url: {data}")


def generate_together(prompt: str, key: str, model: str) -> str | bytes:
    """Generate image via Together AI."""
    model_map = {
        "flux-pro": "black-forest-labs/FLUX.1.1-pro",
        "flux-dev": "black-forest-labs/FLUX.1-dev",
        "flux-schnell": "black-forest-labs/FLUX.1-schnell",
    }
    target_model = model_map.get(model, model) or "black-forest-labs/FLUX.1-schnell"

    url = "https://api.together.xyz/v1/images/generations"
    payload = json.dumps(
        {
            "model": target_model,
            "prompt": prompt,
            "width": 1280,
            "height": 720,
            "steps": 4 if "schnell" in target_model.lower() else 28,
            "n": 1,
            "response_format": "b64_json",
        }
    ).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    results = data.get("data", [])
    if results:
        first = results[0]
        if "b64_json" in first and first["b64_json"]:
            return first["b64_json"]
        if "url" in first and first["url"]:
            return first["url"]
    raise RuntimeError(f"Together AI response did not contain image data: {data}")


def generate_replicate(prompt: str, key: str, model: str) -> str:
    """Generate image via Replicate and return image URL."""
    model_map = {
        "flux-pro": "black-forest-labs/flux-1.1-pro",
        "flux-dev": "black-forest-labs/flux-dev",
        "flux-schnell": "black-forest-labs/flux-schnell",
        "recraft": "recraft-ai/recraft-v3",
    }
    target_model = model_map.get(model, model) or "black-forest-labs/flux-1.1-pro"

    url = f"https://api.replicate.com/v1/models/{target_model}/predictions"
    payload = json.dumps(
        {
            "input": {
                "prompt": prompt,
                "aspect_ratio": "16:9",
                "output_format": "png",
            }
        }
    ).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "wait",
    }

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    # Replicate returns output when "Prefer: wait" completes
    output = data.get("output")
    if isinstance(output, list) and output:
        return output[0]
    if isinstance(output, str):
        return output

    # If polling needed
    poll_url = data.get("urls", {}).get("get")
    if poll_url:
        for _ in range(60):
            time.sleep(2)
            poll_req = urllib.request.Request(
                poll_url, headers={"Authorization": f"Bearer {key}"}
            )
            with urllib.request.urlopen(poll_req, timeout=30) as poll_resp:
                poll_data = json.loads(poll_resp.read().decode("utf-8"))
                if poll_data.get("status") == "succeeded":
                    out = poll_data.get("output")
                    if isinstance(out, list) and out:
                        return out[0]
                    if isinstance(out, str):
                        return out
                elif poll_data.get("status") in ("failed", "canceled"):
                    raise RuntimeError(f"Replicate generation {poll_data.get('status')}")

    raise RuntimeError(f"Replicate response failed to provide output: {data}")


def generate_openai(prompt: str, key: str, model: str = "dall-e-3") -> str:
    """Generate image via OpenAI DALL-E 3 and return image URL."""
    url = "https://api.openai.com/v1/images/generations"
    payload = json.dumps(
        {
            "model": model or "dall-e-3",
            "prompt": prompt,
            "size": "1792x1024",
            "quality": "standard",
            "n": 1,
        }
    ).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    results = data.get("data", [])
    if results and "url" in results[0]:
        return results[0]["url"]
    raise RuntimeError(f"OpenAI response did not contain image url: {data}")


def generate_cloudflare(prompt, output_path):
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")
    model = os.getenv("IMAGE_MODEL", "@cf/black-forest-labs/flux-1-schnell")
    if not model or not model.startswith("@cf/"):
        model = "@cf/black-forest-labs/flux-1-schnell"

    if not account_id or not api_token:
        print("[WARN] Cloudflare credentials not found (CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN required), falling back to local")
        return generate_local(prompt, output_path)

    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    # Cloudflare FLUX.1 [schnell] strictly permits only: prompt (max 2048 chars) and steps (1-8, default 4)
    payload = {
        "prompt": prompt[:2048].strip(),
        "steps": 4
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            print(f"[WARN] Cloudflare API error ({response.status_code}): {response.text}")
            return generate_local(prompt, output_path)

        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            data = response.json()
            if not data.get("success", True) and "result" not in data:
                print(f"[WARN] Cloudflare request unsuccessful: {data.get('errors')}")
                return generate_local(prompt, output_path)
            image_b64 = data.get("result", {}).get("image")
            if not image_b64:
                print("[WARN] Cloudflare response contained no image data, falling back to local")
                return generate_local(prompt, output_path)
            image_bytes = base64.b64decode(image_b64)
        else:
            image_bytes = response.content

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.save(output_path)
        print(f"[OK] Cloudflare image generated: {output_path}")
        return output_path
    except Exception as e:
        print(f"[WARN] Cloudflare failed: {e}")
        return generate_local(prompt, output_path)


def generate_vast(
    prompt: str,
    key: str = "",
    model: str = "",
    width: int = 1280,
    height: int = 720,
    steps: int = 4,
    guidance_scale: float = 1.0,
) -> bytes:
    """Generate image on remote GPU via Vast server endpoint."""
    vast_url = os.environ.get("VAST_SERVER_URL", "").rstrip("/")
    if not vast_url:
        raise RuntimeError("VAST_SERVER_URL is not set. Launch vast GPU first via --vast.")

    url = f"{vast_url}/v1/images/generate"
    payload = json.dumps({
        "prompt": prompt,
        "model": model,
        "width": width,
        "height": height,
        "steps": steps if steps > 0 else 4,
        "guidance_scale": guidance_scale,
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
        if res_data.get("status") == "success" and "image_base64" in res_data:
            return base64.b64decode(res_data["image_base64"])
        elif "error" in res_data:
            raise RuntimeError(f"Vast image generation failed: {res_data['error']}")
    raise RuntimeError(f"Unexpected Vast response: {res_data}")


def generate_image(prompt, output_path):
    provider = os.getenv("IMAGE_PROVIDER", "local").lower()
    if provider == "local":
        return generate_local(prompt, output_path)
    elif provider == "huggingface":
        return generate_huggingface(prompt, output_path)
    elif provider == "fireworks":
        return generate_fireworks(prompt, output_path)
    elif provider == "together":
        return generate_together(prompt, output_path)
    elif provider == "fal":
        return generate_fal(prompt, output_path)
    elif provider == "replicate":
        return generate_replicate(prompt, output_path)
    elif provider == "openai":
        return generate_openai(prompt, output_path)
    elif provider in ("cloudflare", "cf"):
        return generate_cloudflare(prompt, output_path)
    else:
        print(f"[WARN] Unknown image provider: {provider}, falling back to local")
        return generate_local(prompt, output_path)


def generate_local(
    prompt: str,
    output_path: str,
    model: str = "",
    width: int = 1280,
    height: int = 720,
) -> str:
    """Generate image locally via SD/LCM and write to output_path."""
    target_model = model or os.environ.get("IMAGE_MODEL", "segmind/SSD-1B")
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = generate_local_lcm(prompt, model=target_model, width=width, height=height)
        dest.write_bytes(data)
        return str(dest)
    except Exception as e:
        print(f"  [Local Generation Warning] {e}. Generating procedural fallback...")
        try:
            from PIL import Image, ImageDraw
            img = Image.new("RGB", (width, height), color=(20, 24, 30))
            draw = ImageDraw.Draw(img)
            draw.text((width // 4, height // 2), f"Scene: {dest.stem}", fill=(220, 220, 220))
            img.save(dest, format="PNG")
        except Exception:
            dest.write_bytes(b"")
        return str(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AI images for script.json scenes.")
    parser.add_argument("--script", default="script.json", help="Path to script.json")
    parser.add_argument("--out", default="public/images", help="Output directory for images")
    parser.add_argument(
        "--provider",
        "--imageprovider",
        "--image-provider",
        dest="provider",
        help="Image provider (local, free, pollinations, vast, fireworks, together, fal, replicate, openai)",
    )
    parser.add_argument(
        "--model",
        "--imagemodel",
        "--image-model",
        dest="model",
        help="Model name (e.g. segmind/SSD-1B, SimianLuo/LCM_Dreamshaper_v7, flux-schnell, flux-dev, flux-pro, recraft)",
    )
    parser.add_argument("--force", action="store_true", help="Force regeneration of existing images")

    parser.add_argument(
        "--visual-style",
        type=str,
        default=None,
        help="Custom visual style prefix (e.g. 'Cinematic anime cel...')",
    )
    parser.add_argument(
        "--style", "--animation-style",
        dest="animation_style",
        type=str,
        default=None,
        help="2D Animation style preset (e.g. ink_doodle, flat_vector, documentary, stickman, dark_cinematic, anime_cel, whiteboard, vintage_retro)",
    )
    parser.add_argument(
        "--styles",
        action="store_true",
        help="List all supported 2D animation styles and exit",
    )

    args = parser.parse_args()

    if getattr(args, "styles", False):
        print("\n=== Supported 2D Animation Styles ===")
        for s in list_animation_styles():
            print(f"  • {s['key']:<15} | {s['name']}")
            print(f"    Description : {s['description']}")
            print(f"    Best For    : {s['best_for']}\n")
        return

    generate_images_for_script(
        script=args.script,
        output_dir=args.out,
        provider=getattr(args, "provider", None),
        model=getattr(args, "model", None),
        force=getattr(args, "force", False),
        visual_style=args.visual_style,
        animation_style=getattr(args, "animation_style", None),
    )

    print(f"Generated images for {args.script}")


def generate_images_for_script(
    script: dict | str | Path,
    output_dir: str | Path = "public/images",
    provider: str | None = None,
    model: str | None = None,
    force: bool = False,
    visual_style: str | None = None,
    animation_style: str | None = None,
    **kwargs,
) -> dict:
    """Generate still images for each scene in script.json."""
    root_dir = Path(__file__).resolve().parent.parent
    if isinstance(script, (str, Path)):
        script_path = Path(script)
        script_data = json.loads(script_path.read_text(encoding="utf-8"))
    else:
        script_path = None
        script_data = script

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prov, key, default_model = detect_provider(provider)
    effective_model = model or default_model
    topic = script_data.get("topic", "")
    genre = script_data.get("genre", "documentary")
    width = int(script_data.get("width") or 1920)
    height = int(script_data.get("height") or 1080)
    vis_style = visual_style or script_data.get("visualStyle")

    scenes = script_data.get("scenes", [])
    for idx, scene in enumerate(scenes):
        scene_id = scene.get("id") or f"scene-{idx + 1:03d}"
        dest = out_dir / f"{scene_id}.png"

        rel_path = str(dest.relative_to(root_dir).as_posix()) if root_dir in dest.parents else str(dest.as_posix())
        if dest.exists() and not force:
            scene["image"] = rel_path
            continue

        prompt = make_prompt(
            scene,
            topic=topic,
            genre=genre,
            index=idx,
            visual_style=vis_style,
            animation_style=animation_style or scene.get("animationStyle"),
            theme=script_data.get("theme"),
            characters_dict=script_data.get("characters"),
        )

        bg_prompt = scene.get("backgroundPrompt") or ""
        char_prompt = scene.get("characterPrompt") or ""
        print(f"\n  [Image Gen] {scene_id}:")
        print(f"    Background : {bg_prompt}")
        print(f"    Characters : {char_prompt}")

        try:
            if prov == "fetch":
                queries = build_search_queries(scene, topic=topic, genre=genre)
                data = fetch_free_related_image(queries, width=width, height=height)
                save_image_data(data, dest)
            elif prov == "vast":
                data = generate_vast(prompt, key=key, model=effective_model, width=width, height=height)
                save_image_data(data, dest)
            elif prov == "pollinations":
                data = generate_pollinations(prompt, model=effective_model, api_key=key, width=width, height=height)
                save_image_data(data, dest)
            elif prov == "huggingface":
                data = generate_huggingface(prompt, key=key, model=effective_model, width=width, height=height)
                save_image_data(data, dest)
            elif prov == "fireworks":
                data = generate_fireworks(prompt, key=key, model=effective_model)
                save_image_data(data, dest)
            elif prov == "together":
                data = generate_together(prompt, key=key, model=effective_model)
                save_image_data(data, dest)
            elif prov == "fal":
                url = generate_fal(prompt, key=key, model=effective_model)
                save_image_data(url, dest)
            elif prov == "replicate":
                url = generate_replicate(prompt, key=key, model=effective_model)
                save_image_data(url, dest)
            elif prov == "openai":
                url = generate_openai(prompt, key=key, model=effective_model)
                save_image_data(url, dest)
            elif prov in ("cloudflare", "cf"):
                generate_cloudflare(prompt, str(dest))
            else:  # local diffusers / SD
                data = generate_local_lcm(prompt, model=effective_model, steps=kwargs.get("steps", 2), width=width, height=height)
                save_image_data(data, dest)

            scene["image"] = rel_path
            print(f"  [Image] {scene_id} -> {dest.name}")
        except Exception as exc:
            print(f"  [Image Error] Failed to generate {scene_id} ({exc}). Using procedural fallback.")
            try:
                from PIL import Image, ImageDraw
                dest.parent.mkdir(parents=True, exist_ok=True)
                img = Image.new("RGB", (width, height), color=(20, 24, 30))
                draw = ImageDraw.Draw(img)
                draw.text((width // 4, height // 2), f"Scene {idx + 1}: {scene_id}", fill=(220, 220, 220))
                img.save(dest, format="PNG")
                scene["image"] = rel_path
            except Exception:
                pass

    if script_path:
        script_path.write_text(json.dumps(script_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return script_data


generate_images = generate_images_for_script
generate_scene_images = generate_images_for_script

if __name__ == "__main__":
    main()
