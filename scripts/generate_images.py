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
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


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


STYLE_PROMPT_PREFIX = (
    "Cinematic horror masterpiece, hyper-realistic eerie 2D digital painting, "
    "intense dark moody lighting, thick volumetric fog, deep unsettling shadows, "
    "ominous mystery, high contrast, terrifying atmosphere: "
)

STYLE_PROMPT_SUFFIX = (
    ", highly detailed background, cinematic composition, widescreen 16:9, "
    "dramatic lighting, sharp focus, 8k resolution, masterpiece."
)


def normalize_provider_name(name: str | None) -> str:
    """Normalize provider aliases to canonical names."""
    if not name:
        return ""
    n = name.lower().strip().replace("-", "").replace("_", "").replace(" ", "")
    if n in ("fetch", "stock", "free", "freeimage", "freeimages", "search", "unsplash", "lexica", "wiki"):
        return "fetch"
    if n in ("local", "lcm", "sd", "sd15", "localdiffusers", "sdcpp", "openvino", "onnx", "dreamshaper"):
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


def extract_script_visual_keywords(scene: dict, topic: str = "", genre: str | None = None) -> list[str]:
    """Extract script-specific visual entities prioritizing THIS SCENE's unique content."""
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

    # Default fallback if empty
    if not scene_keywords:
        if gp:
            scene_keywords.append(gp.default_keywords)
        else:
            scene_keywords.append("cinematic landscape atmosphere background")

    return scene_keywords


def build_search_queries(scene: dict, topic: str = "", genre: str | None = None) -> list[str]:
    """Build a prioritized list of diverse search queries tailored to THIS scene."""
    try:
        import beats
        gp = beats.detect_genre(topic, genre or scene.get("genre"))
    except Exception:
        gp = None

    raw_keywords = extract_script_visual_keywords(scene, topic, genre)
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


def make_search_keywords(scene: dict, topic: str = "", genre: str | None = None) -> str:
    """Build clean, targeted English search keywords for stock/related image retrieval."""
    queries = build_search_queries(scene, topic, genre)
    return queries[0] if queries else "cinematic atmospheric background"


def make_prompt(
    scene: dict,
    topic: str = "",
    genre: str | None = None,
    visual_style: str | None = None,
    index: int = 0,
) -> str:
    """Construct an evocative visual prompt in English tailored uniquely to this scene."""
    try:
        import beats
        gp = beats.detect_genre(topic, genre or scene.get("genre"))
    except Exception:
        gp = None

    beat = scene.get("beat", "").strip()
    emotion = scene.get("emotion", "").strip()
    headline = scene.get("headline", "").strip()

    # Extract scene-specific visual cues
    extracted_keywords = extract_script_visual_keywords(scene, topic, genre)
    scene_entities = list(dict.fromkeys(" ".join(extracted_keywords).split()))[:7]
    scene_subject = " ".join(scene_entities) if scene_entities else "atmospheric environment"

    # Extract shot visual types
    visual_words = []
    shots = scene.get("shots", [])
    for s in shots:
        v = s.get("visual", {})
        v_type = v.get("type", "")
        if v_type and v_type not in ("image", "black", "void"):
            visual_words.append(v_type)
            if v.get("lit"):
                visual_words.append("lit window warm illumination")
            if v.get("flicker"):
                visual_words.append("flickering flame light")
            if v.get("glow"):
                visual_words.append("glowing magical radiance")
            if v.get("lightning"):
                visual_words.append("lightning storm flashes")

    context_parts = [scene_subject]
    if visual_words:
        unique_visuals = list(dict.fromkeys(visual_words))
        context_parts.append(f"featuring {', '.join(unique_visuals)}")
    if beat:
        context_parts.append(f"{beat} scene")
    if emotion:
        context_parts.append(f"{emotion} mood")

    core = ", ".join(context_parts)
    prefix = visual_style or (gp.visual_style_prefix if gp else STYLE_PROMPT_PREFIX)
    suffix = gp.visual_style_suffix if gp else STYLE_PROMPT_SUFFIX
    return f"{prefix}{core}{suffix}"


def detect_provider(requested_provider: str | None = None) -> tuple[str, str, str]:
    """Detect available image provider, key, and default model.

    Returns (provider_name, api_key, default_model).
    Defaults to local Diffusers generation ('local', 'segmind/SSD-1B').
    """
    raw_env_provider = os.environ.get("IMAGE_PROVIDER", "").strip()
    provider = normalize_provider_name(requested_provider or raw_env_provider)
    model = os.environ.get("IMAGE_MODEL", "").strip()

    if provider in ("fetch", "free", "stock", "unsplash", "lexica"):
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


def fetch_free_related_image(scene: dict, topic: str = "", index: int = 0, genre: str | None = None) -> bytes:
    """Fetch matching high-res related image for free via Lexica / Unsplash / Wikimedia / Openverse with Pollinations fallback."""
    queries = build_search_queries(scene, topic, genre=genre)
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
    """Check if model name refers to Segmind SSD-1B or default local pipeline."""
    if not model_name:
        return True
    m = model_name.lower().strip()
    return "ssd" in m or "ssd-1b" in m or "segmind" in m or m in ("default", "local", "")


def is_flux_model(model_name: str | None) -> bool:
    """Check if model name refers to FLUX model."""
    if not model_name:
        return False
    m = model_name.lower().strip()
    return "flux" in m


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

    # 3. Handle FLUX if specified
    elif is_flux_model(target):
        print(f"  [Local FLUX] Initializing FLUX pipeline for '{target}'...")
        try:
            from diffusers import Flux2KleinPipeline
            base_model_id = "black-forest-labs/FLUX.2-klein-4B"
            pipe = Flux2KleinPipeline.from_pretrained(
                base_model_id,
                torch_dtype=torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float32,
            )
            _LOCAL_DIFFUSION_PIPELINE = pipe
        except Exception as e:
            print(f"  [Local FLUX] FLUX load notice ({e}). Falling back to optimized LCM / SD pipeline...")
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
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        for ep in endpoints:
            try:
                req = urllib.request.Request(ep, data=payload, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=120) as resp:
                    res_data = json.loads(resp.read().decode("utf-8"))
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

    # SDXL-Turbo requires 1 step and guidance_scale=0.0 (or <=1.0)
    if is_turbo:
        inf_steps = 1
        cfg = 0.0
    elif is_flux:
        inf_steps = steps if (steps and steps > 0) else 3
        cfg = 0.0
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


def download_image(url: str, dest: Path) -> None:
    """Download image binary to destination path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "tube2dscript/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def generate_images_for_script(
    script_data: dict,
    *,
    out_dir: str | Path = "public/images",
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    force: bool = False,
) -> dict:
    """Generate background images for each scene in script_data and return updated dict."""
    det_provider, det_key, det_model = detect_provider(provider)
    resolved_provider = normalize_provider_name(provider) or det_provider or "fetch"
    api_key = api_key or det_key
    model = model or det_model

    if resolved_provider not in ("pollinations", "fetch") and (not api_key or api_key.startswith("...")):
        print(
            f"WARNING: Image provider '{resolved_provider}' requested, but valid API key is not set.\n"
            f"Falling back to pre-generated Free Related Image Search (Lexica / Unsplash / Stock).",
            file=sys.stderr,
        )
        resolved_provider = "fetch"
        api_key = "free"
        model = "free-search"

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    topic = script_data.get("topic", script_data.get("title", ""))
    genre = script_data.get("genre")
    visual_style = script_data.get("visualStyle")
    scenes = script_data.get("scenes", [])
    total_scenes = len(scenes)

    width = script_data.get("width") or 1920
    height = script_data.get("height") or 1080
    aspect_ratio = script_data.get("aspectRatio") or ("9:16" if height > width else "1:1" if height == width else "16:9")
    sd_latent_width = script_data.get("sdLatentWidth")
    sd_latent_height = script_data.get("sdLatentHeight")

    if resolved_provider == "fetch":
        provider_title = "Free Related Image Search (Lexica / Unsplash / Stock)"
    elif resolved_provider == "local":
        if is_ssd_model(model):
            provider_title = f"Local SSD-1B + LCM-LoRA ({aspect_ratio} @ {width}x{height})"
        elif is_turbo_model(model):
            provider_title = f"Local SDXL-Turbo ({aspect_ratio} @ {width}x{height})"
        elif is_flux_model(model):
            provider_title = f"Local FLUX ({aspect_ratio} @ {width}x{height})"
        else:
            provider_title = f"Local SD 1.5 + LCM-LoRA ({aspect_ratio} @ {width}x{height})"
    elif resolved_provider == "pollinations":
        provider_title = f"Pollinations AI (FREE, {aspect_ratio} @ {width}x{height})"
    else:
        provider_title = f"{resolved_provider.upper()} ({aspect_ratio} @ {width}x{height})"
    print(f"Retrieving background images via {provider_title} ({model or 'default'}) for {total_scenes} scenes...")

    burns_modes = ["in", "out", "left", "right"]

    for i, scene in enumerate(scenes):
        scene_id = scene.get("id", f"scene-{i+1:03d}")
        shots = scene.get("shots", [])
        if not shots:
            continue

        prompt = make_prompt(scene, topic, genre=genre, visual_style=visual_style, index=i)
        image_filename = f"{scene_id}.png"
        image_dest = out_path / image_filename
        rel_static_path = f"images/{image_filename}"

        if not force and image_dest.is_file() and image_dest.stat().st_size > 1000:
            print(f"  [{i+1}/{total_scenes}] Scene {i+1}: cached ({image_filename})")
        else:
            scene_label = scene.get("headline", "") or f"Scene {i+1}"
            if resolved_provider == "fetch":
                query_preview = make_search_keywords(scene, topic, genre=genre)
                print(f"  [{i+1}/{total_scenes}] Fetching matching image for {scene_label[:30]} (query: '{query_preview[:45]}')...")
            else:
                print(f"  [{i+1}/{total_scenes}] Generating image for {scene_label[:30]} ({aspect_ratio})...")

            try:
                if resolved_provider == "fetch":
                    img_data = fetch_free_related_image(scene, topic, index=i, genre=genre)
                elif resolved_provider == "local":
                    # Deterministic scene seed
                    scene_seed = int(hashlib.md5(f"{topic}_{scene_id}_{i}".encode("utf-8")).hexdigest()[:8], 16)
                    img_data = generate_local_lcm(
                        prompt,
                        model=model,
                        seed=scene_seed,
                        width=width,
                        height=height,
                        sd_width=sd_latent_width,
                        sd_height=sd_latent_height,
                    )
                elif resolved_provider == "pollinations":
                    if i > 0:
                        time.sleep(1.5)  # Pace free requests
                    # Deterministic scene seed
                    scene_seed = int(hashlib.md5(f"{topic}_{scene_id}_{i}".encode("utf-8")).hexdigest()[:8], 16)
                    img_data = generate_pollinations(
                        prompt,
                        model,
                        api_key=api_key,
                        seed=scene_seed,
                        width=width,
                        height=height,
                    )
                elif resolved_provider in ("huggingface", "hf", "gradio"):
                    try:
                        img_data = generate_gradio_hf(prompt, api_key, model=model, width=width, height=height)
                    except Exception as hf_err:
                        print(f"  [HF Gradio Failover] Hugging Face Space generation failed ({hf_err}). Switching automatically to local SSD-1B / SDXL-Turbo on CPU...")
                        scene_seed = int(hashlib.md5(f"{topic}_{scene_id}_{i}".encode("utf-8")).hexdigest()[:8], 16)
                        try:
                            img_data = generate_local_lcm(
                                prompt,
                                model="segmind/SSD-1B",
                                seed=scene_seed,
                                width=width,
                                height=height,
                                sd_width=sd_latent_width,
                                sd_height=sd_latent_height,
                            )
                        except Exception as local_err:
                            print(f"  [Local SSD-1B Failover] Local SSD-1B failed ({local_err}). Trying local SDXL-Turbo...")
                            try:
                                img_data = generate_local_lcm(
                                    prompt,
                                    model="stabilityai/sdxl-turbo",
                                    seed=scene_seed,
                                    width=width,
                                    height=height,
                                    sd_width=sd_latent_width,
                                    sd_height=sd_latent_height,
                                )
                            except Exception as turbo_err:
                                print(f"  [Failover to Stock Search] Local generation failed ({turbo_err}). Using free related image search...")
                                img_data = fetch_free_related_image(scene, topic, index=i, genre=genre)
                elif resolved_provider == "fireworks":
                    img_data = generate_fireworks(prompt, api_key, model)
                elif resolved_provider == "together":
                    img_data = generate_together(prompt, api_key, model)
                elif resolved_provider == "fal":
                    img_data = generate_fal(prompt, api_key, model)
                elif resolved_provider == "replicate":
                    img_data = generate_replicate(prompt, api_key, model)
                elif resolved_provider == "openai":
                    img_data = generate_openai(prompt, api_key, model)
                else:
                    raise ValueError(f"Unknown image provider: {resolved_provider}")

                save_image_data(img_data, image_dest)
                print(f"       -> Saved {image_filename}")
            except Exception as err:
                print(f"       ! Error retrieving image for Scene {i+1}: {err}", file=sys.stderr)
                continue

        # Update visual in shots to use the new image
        for s_idx, shot in enumerate(shots):
            burn = burns_modes[(i + s_idx) % len(burns_modes)]
            shot["visual"] = {
                "type": "image",
                "src": rel_static_path,
                "kenBurns": burn,
                "dim": 0.15,
                "prompt": prompt,
            }

    if resolved_provider == "local" or ("_LOCAL_PIPE" in globals() and _LOCAL_PIPE is not None):
        cleanup_local_pipeline()

    return script_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AI images for script.json scenes.")
    parser.add_argument("--script", default="script.json", help="Path to script.json")
    parser.add_argument("--out", default="public/images", help="Output directory for images")
    parser.add_argument(
        "--provider",
        "--imageprovider",
        "--image-provider",
        dest="provider",
        help="Image provider (local, free, pollinations, fireworks, together, fal, replicate, openai)",
    )
    parser.add_argument(
        "--model",
        "--imagemodel",
        "--image-model",
        dest="model",
        help="Model name (e.g. segmind/SSD-1B, SimianLuo/LCM_Dreamshaper_v7, flux-schnell, flux-dev, flux-pro, recraft)",
    )
    parser.add_argument("--force", action="store_true", help="Force regeneration of existing images")

    args = parser.parse_args()
    script_file = Path(args.script)
    if not script_file.is_file():
        sys.exit(f"Error: Script file '{args.script}' not found.")

    with open(script_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated = generate_images_for_script(
        data,
        out_dir=args.out,
        provider=args.provider,
        model=args.model,
        force=args.force,
    )

    with open(script_file, "w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)

    print(f"Updated {args.script} with generated images.")


if __name__ == "__main__":
    main()
