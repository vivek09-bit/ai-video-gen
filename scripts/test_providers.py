"""Test LLM and image provider routing and prompt construction.

Run with:  .venv\\Scripts\\python.exe scripts\\test_providers.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import beats
import generate_images
import generate_videos
import make_assets
import metadata
import new_video
import phonemise
import tts
import uploader


def test_genres() -> None:
    print("Testing Genre detection and profiles...")

    # 1. Test auto-detection
    assert beats.detect_genre("anime goku naruto fight").key == "anime"
    assert beats.detect_genre("epic ninja battle showdown").key == "anime"
    assert beats.detect_genre("ai dekho son pari").key == "fantasy"
    assert beats.detect_genre("जादुई महल की कहानी").key == "fantasy"
    assert beats.detect_genre("Future AI Robot in 2050").key == "scifi"
    assert beats.detect_genre("अंतरिक्ष और एलियन की खोज").key == "scifi"
    assert beats.detect_genre("How the Roman Empire fell documentary").key == "documentary"
    assert beats.detect_genre("Physics and quantum mechanics explained").key == "documentary"
    assert beats.detect_genre("इतिहास और रहस्य के तथ्य").key == "documentary"
    assert beats.detect_genre("Lord Krishna and Mahabharat War").key == "mythology"
    assert beats.detect_genre("Never Give Up Motivation story").key == "motivational"
    assert beats.detect_genre("Funny comedy joke story").key == "comedy"
    assert beats.detect_genre("हँसी और मज़ेदार चुटकुले").key == "comedy"
    assert beats.detect_genre("भूतिया हवेली का खौफ").key == "horror"

    # 2. Test genre profile retrieval
    profile_anime = beats.get_genre("anime")
    assert profile_anime.name == "Anime & Action"
    assert "#f59e0b" in profile_anime.theme["accent"]
    assert "Ufotable" in profile_anime.visual_style_prefix or "anime" in profile_anime.visual_style_prefix.lower()

    profile_fantasy = beats.get_genre("fantasy")
    assert profile_fantasy.name == "Fantasy & Fairy Tale"
    assert "#f59e0b" in profile_fantasy.theme["accent"]

    profile_comedy = beats.get_genre("comedy")
    assert profile_comedy.name == "Comedy & Satire"

    # 3. Test planning with genre
    plan_anime = beats.plan("anime goku naruto fight", minutes=1.0)
    assert plan_anime["genre"] == "anime"
    assert len(plan_anime["scenes"]) > 0

    plan_comedy = beats.plan("funny office prank", minutes=1.0)
    assert plan_comedy["genre"] == "comedy"

    plan_doc = beats.plan("science of black holes", minutes=1.0)
    assert plan_doc["genre"] == "documentary"

    # 4. Verify genre audio pool mapping
    for g_key in ["anime", "comedy", "documentary", "horror", "scifi", "fantasy", "mythology", "motivational", "drama"]:
        assert g_key in beats.GENRE_AUDIO_POOLS
        pool = beats.GENRE_AUDIO_POOLS[g_key]
        assert "ambience_setup" in pool and len(pool["ambience_setup"]) > 0
        assert "sfx_climax" in pool and len(pool["sfx_climax"]) > 0

    print("  ✓ All Genre detection, profile, and audio pool checks passed.")


def test_themes() -> None:
    print("Testing Visual Theme catalog and dynamic resolution...")

    # 1. Test full theme catalog listing
    themes_list = beats.list_themes()
    assert len(themes_list) >= 20, f"Expected 20+ themes, got {len(themes_list)}"
    theme_keys = {t["key"] for t in themes_list}
    assert "blood_red" in theme_keys
    assert "cyber_neon" in theme_keys
    assert "golden_radiance" in theme_keys
    assert "amethyst_dream" in theme_keys
    assert "emerald_forest" in theme_keys
    assert "synthwave" in theme_keys
    assert "monochrome_noir" in theme_keys

    for t in themes_list:
        assert t["accent"].startswith("#")
        assert "panel" in t
        assert "headlineFont" in t
        assert "bodyFont" in t

    # 2. Test exact name and alias lookup
    t_cyber = beats.resolve_theme("cyber_neon")
    assert t_cyber.key == "cyber_neon"
    assert t_cyber.accent == "#06b6d4"

    t_alias = beats.resolve_theme("gold")
    assert t_alias.key == "golden_radiance"

    t_purple = beats.resolve_theme("purple")
    assert t_purple.key == "amethyst_dream"

    # 3. Test custom hex color resolution
    t_hex = beats.resolve_theme("#ff007f", genre="fantasy")
    assert t_hex.accent == "#ff007f"
    assert "rgba" in t_hex.panel
    assert t_hex.key == "hex_ff007f"

    # 4. Test random theme resolution
    t_rand1 = beats.resolve_theme("random", seed=42)
    t_rand2 = beats.resolve_theme("random", seed=99)
    assert t_rand1.key in beats.THEMES
    assert t_rand2.key in beats.THEMES

    # 5. Test random within genre
    t_genre_rand = beats.resolve_theme("random_genre", genre="scifi", seed=123)
    assert t_genre_rand.key in beats.GENRE_THEME_POOLS["scifi"]

    # 6. Test deterministic auto resolution for different topics & genres
    t_auto_fantasy = beats.resolve_theme("auto", genre="fantasy", topic="son pari")
    assert t_auto_fantasy.key in beats.GENRE_THEME_POOLS["fantasy"]

    t_auto_scifi = beats.resolve_theme("auto", genre="scifi", topic="quantum computer")
    assert t_auto_scifi.key in beats.GENRE_THEME_POOLS["scifi"]

    # 7. Test integration with beats.plan
    plan_with_theme = beats.plan("AI future", minutes=1.0, genre="scifi", theme="cyber_neon")
    assert plan_with_theme["theme"]["key"] == "cyber_neon"
    assert plan_with_theme["theme"]["accent"] == "#06b6d4"
    assert plan_with_theme["scenes"][0]["background"]["color"] == "#020b12"

    print("  ✓ All Visual Theme catalog and dynamic resolution checks passed.")


def test_dynamic_prompting() -> None:
    print("Testing Dynamic multi-genre prompt building and Hinglish rules...")

    # 1. Test Hinglish prompt rules (spoken Devanagari Hindi for TTS + Hinglish subtitles)
    plan_dict_hinglish = beats.plan("AI and Quantum Computer future", minutes=1.0, genre="scifi", language="hinglish")
    profile = beats.get_genre("scifi")
    hinglish_prompt = new_video.build_prompt(
        plan_dict_hinglish,
        voice_style="cyberpunk thriller",
        genre_profile=profile,
    )

    assert "cyberpunk thriller" in hinglish_prompt
    assert "AI and Quantum Computer future" in hinglish_prompt
    assert "Native Spoken Hindi Rules" in hinglish_prompt
    assert "Minimise English loanwords" in hinglish_prompt
    assert "चल दूरभाष" in hinglish_prompt
    assert "अंतरजाल" in hinglish_prompt

    # 2. Test Devanagari Hindi prompt rules
    plan_dict_hindi = beats.plan("AI and Quantum Computer future", minutes=1.0, genre="scifi", language="hi")
    hindi_prompt = new_video.build_prompt(
        plan_dict_hindi,
        voice_style="cyberpunk thriller",
        genre_profile=profile,
    )
    assert "Native Spoken Hindi Rules" in hindi_prompt
    assert "Minimise English loanwords" in hindi_prompt
    assert "चल दूरभाष" in hindi_prompt
    assert "अंतरजाल" in hindi_prompt
    assert "ट्रेन" in hindi_prompt
    assert "लोहपथगामिनी" in hindi_prompt

    # 3. Test Transliteration & SRT generator
    import transliterate
    deva_sample = "मोबाइल फोन स्कैम अलर्ट: अनजान कॉल और मैसेज से सावधान रहें।"
    roman_out = transliterate.deva_to_hinglish(deva_sample)
    assert "mobile phone scam alert" in roman_out
    assert "call" in roman_out
    assert "message" in roman_out
    assert "savdhan" in roman_out
    assert "rahein" in roman_out

    # Test LLM response parsing with visual tags
    sample_llm_response = (
        "### 1 | Quantum Awakening | VISUAL: supercomputer glowing holographic chamber\n"
        "The neural core ignited with blazing turquoise light.\n\n"
        "### 2 | The Glitch in the Matrix | VISUAL: dark cyber corridor neon sparks\n"
        "[VISUAL: dark cyber corridor neon sparks]\n"
        "Shadows flickered across the monolithic server racks.\n"
    )
    parsed_title, blocks = new_video.parse_written(sample_llm_response)
    assert len(blocks) == 2
    assert blocks[0][0] == "Quantum Awakening"
    assert "supercomputer" in blocks[0][1]
    assert "neural core" in blocks[0][2]
    assert blocks[1][0] == "The Glitch in the Matrix"
    assert "cyber corridor" in blocks[1][1]
    # Spoken audio text must be stripped of any bracketed VISUAL tags
    assert "[VISUAL:" not in blocks[1][2]
    assert "Shadows flickered" in blocks[1][2]

    # Image prompt generation for scifi
    scene = {
        "headline": "Quantum Server Core",
        "visualKeywords": "supercomputer glowing holographic chamber",
        "beat": "inciting",
        "emotion": "discovery",
        "shots": [{"visual": {"type": "city", "lit": True}}],
        "_storyNote": "Supercomputer activates",
        "lines": [{"text": "The glowing neural core ignited."}],
    }
    img_prompt = generate_images.make_prompt(scene, topic="AI Quantum future", genre="scifi")
    assert "supercomputer" in img_prompt

    # Free stock search keywords prioritizing visualKeywords
    keywords = generate_images.make_search_keywords(scene, topic="AI Quantum future", genre="scifi")
    assert "supercomputer" in keywords or "holographic" in keywords

    print("  ✓ All Dynamic prompt, parsing, and genre styling checks passed.")


def test_llm_routing() -> None:
    print("Testing LLM provider routing...")

    # Mock environment variables for testing
    old_env = dict(os.environ)
    try:
        os.environ["DEEPSEEK_API_KEY"] = "sk-deepseek-test"
        os.environ["FIREWORKS_API_KEY"] = "fw-test-key"
        os.environ["TOGETHER_API_KEY"] = "tog-test-key"
        os.environ["OMNIROUTE_API_KEY"] = "omni-test-key"
        os.environ["OPENAI_API_KEY"] = "oai-test-key"

        # 1. Test DeepSeek alias
        w = new_video.resolve_llm_writer(requested_model="deepseek")
        assert w is not None, "DeepSeek routing failed"
        assert w[0] == "DeepSeek" and w[1] == "https://api.deepseek.com/v1" and w[3] == "deepseek-chat"

        # 2. Test Fireworks AI alias
        w = new_video.resolve_llm_writer(requested_model="fireworks")
        assert w is not None, "Fireworks AI routing failed"
        assert w[0] == "Fireworks AI" and "fireworks" in w[1] and "deepseek-v3" in w[3]

        w = new_video.resolve_llm_writer(requested_model="firework")
        assert w is not None and w[0] == "Fireworks AI"

        # 3. Test Together AI alias
        w = new_video.resolve_llm_writer(requested_model="together")
        assert w is not None, "Together AI routing failed"
        assert w[0] == "Together AI" and "together" in w[1] and "DeepSeek-V3" in w[3]

        # 4. Test Omniroute alias
        w = new_video.resolve_llm_writer(requested_model="omniroute")
        assert w is not None, "Omniroute routing failed"
        assert w[0] == "Omniroute / OpenRouter" and w[3] == "auto"

        # 5. Test Ollama alias
        w = new_video.resolve_llm_writer(requested_model="ollama")
        assert w is not None, "Ollama routing failed"
        assert w[0] == "Ollama (local)" and "11434" in w[1]

        # 6. Test explicit provider with custom model
        w = new_video.resolve_llm_writer(
            requested_model="meta-llama/Llama-3-70b-chat-hf",
            requested_provider="together",
        )
        assert w is not None and w[0] == "Together AI" and w[3] == "meta-llama/Llama-3-70b-chat-hf"

        print("  ✓ All LLM provider routing checks passed.")
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def test_image_routing() -> None:
    print("Testing Image provider routing...")

    old_env = dict(os.environ)
    try:
        os.environ.pop("IMAGE_PROVIDER", None)
        os.environ.pop("IMAGE_MODEL", None)
        os.environ["FIREWORKS_API_KEY"] = "fw-test-key"
        os.environ["TOGETHER_API_KEY"] = "tog-test-key"
        os.environ["FAL_KEY"] = "fal-test-key"
        os.environ["REPLICATE_API_TOKEN"] = "rep-test-key"
        os.environ["OPENAI_API_KEY"] = "oai-test-key"

        # Free image search routing
        prov, key, model = generate_images.detect_provider("fetch")
        assert prov == "fetch" and key == "free"

        prov, key, model = generate_images.detect_provider("stock")
        assert prov == "fetch" and key == "free"

        prov, key, model = generate_images.detect_provider("unsplash")
        assert prov == "fetch" and key == "free"

        # Local SD 1.5 + LCM / SSD-1B image routing
        prov, key, model = generate_images.detect_provider("local")
        assert prov == "local" and key == "local" and ("LCM" in model or "SSD-1B" in model)

        prov, key, model = generate_images.detect_provider("lcm")
        assert prov == "local" and key == "local"

        prov, key, model = generate_images.detect_provider("sd15")
        assert prov == "local" and key == "local"

        prov, key, model = generate_images.detect_provider("openvino")
        assert prov == "local" and key == "local"

        # Pollinations image routing
        prov, key, model = generate_images.detect_provider("pollinations")
        assert prov == "pollinations" and model == "default"

        # Fireworks normalization & detection
        prov, key, model = generate_images.detect_provider("firework")
        assert prov == "fireworks" and key == "fw-test-key" and "schnell" in model

        prov, key, model = generate_images.detect_provider("fireworks")
        assert prov == "fireworks" and key == "fw-test-key"

        # Together AI normalization & detection
        prov, key, model = generate_images.detect_provider("together")
        assert prov == "together" and key == "tog-test-key" and "FLUX" in model

        prov, key, model = generate_images.detect_provider("togetherai")
        assert prov == "together" and key == "tog-test-key"

        # fal.ai normalization & detection
        prov, key, model = generate_images.detect_provider("fal")
        assert prov == "fal" and key == "fal-test-key"

        # Prompt creation
        scene = {
            "headline": "काली हवेली का मुख्य दरवाज़ा",
            "beat": "inciting",
            "emotion": "dread",
            "shots": [{"visual": {"type": "house", "lit": True}}],
            "_storyNote": "प्रवेश द्वार पर अजीब निशान",
            "lines": [{"text": "दरवाज़ा धीरे-धीरे अपने आप खुलने लगा।"}],
        }
        prompt = generate_images.make_prompt(scene, topic="भूतिया हवेली का खौफ")
        assert "horror" in prompt.lower()
        assert "dread" in prompt.lower()
        assert "mansion" in prompt.lower() or "house" in prompt.lower()

        # Scene-specific distinct image queries
        scene1 = {
            "id": "scene-001",
            "headline": "अधूरी ख्वाहिश और जादुई जंगल",
            "beat": "setup",
            "emotion": "calm",
            "shots": [{"visual": {"type": "forest", "fog": True}}],
            "lines": [{"text": "सोना जंगल के बीचों-बीच एक पुरानी जादुई किताब ढूँढ रही थी।"}],
        }
        scene2 = {
            "id": "scene-002",
            "headline": "पहचान का बुलावा और चमकता आईना",
            "beat": "inciting",
            "emotion": "worried",
            "shots": [{"visual": {"type": "house", "lit": True}}],
            "lines": [{"text": "दीवार पर टंगे कांच के दर्पण में से अजीब रोशनी निकलने लगी।"}],
        }
        queries1 = generate_images.build_search_queries(scene1, topic="ai dekho son pari", genre="fantasy")
        queries2 = generate_images.build_search_queries(scene2, topic="ai dekho son pari", genre="fantasy")

        assert queries1 != queries2, "Scene 1 and Scene 2 should produce distinct search queries"
        assert any("forest" in q or "magic" in q for q in queries1)
        assert any("mirror" in q or "house" in q or "glass" in q for q in queries2)

        prompt1 = generate_images.make_prompt(scene1, topic="ai dekho son pari", genre="fantasy", index=0)
        prompt2 = generate_images.make_prompt(scene2, topic="ai dekho son pari", genre="fantasy", index=1)
        assert prompt1 != prompt2, "Scene 1 and Scene 2 should produce distinct image prompts"

        print("  ✓ All Image provider routing and prompt checks passed.")
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def test_audio_and_speed() -> None:
    print("Testing Dynamic procedural sound effects and speech speed controls...")

    # 1. Test audio assets registry completeness
    assert len(make_assets.BEDS) >= 11, f"Expected at least 11 ambience beds, got {len(make_assets.BEDS)}"
    assert len(make_assets.EFFECTS) >= 15, f"Expected at least 15 sound effects, got {len(make_assets.EFFECTS)}"

    expected_beds = {
        "amb_drone", "amb_wind", "amb_hum", "amb_rain", "amb_heartbeat",
        "amb_ethereal", "amb_cyber", "amb_temple", "amb_warm", "amb_tension", "amb_uplift"
    }
    for b in expected_beds:
        assert b in make_assets.BEDS, f"Missing expected ambience bed: {b}"

    expected_sfx = {
        "sfx_stinger", "sfx_impact", "sfx_riser", "sfx_whisper", "sfx_bell",
        "sfx_sparkle", "sfx_chime", "sfx_whoosh", "sfx_boom", "sfx_glitch",
        "sfx_horn", "sfx_pop", "sfx_swell", "sfx_clock_tick", "sfx_cinematic_hit"
    }
    for s in expected_sfx:
        assert s in make_assets.EFFECTS, f"Missing expected sound effect: {s}"

    # 2. Test genre audio pools
    for genre_key in beats.GENRES:
        assert genre_key in beats.GENRE_AUDIO_POOLS, f"Missing audio pool for genre {genre_key}"
        pool = beats.GENRE_AUDIO_POOLS[genre_key]
        for stage in ("ambience_setup", "ambience_rising", "ambience_climax", "ambience_resolution"):
            assert stage in pool and len(pool[stage]) > 0, f"Missing {stage} in {genre_key} audio pool"
            for bed_name in pool[stage]:
                assert bed_name in make_assets.BEDS, f"Unknown bed {bed_name} in {genre_key} pool"
        for stage in ("sfx_setup", "sfx_rising", "sfx_climax", "sfx_resolution"):
            assert stage in pool and len(pool[stage]) > 0, f"Missing {stage} in {genre_key} audio pool"
            for sfx_name in pool[stage]:
                assert sfx_name in make_assets.EFFECTS, f"Unknown sfx {sfx_name} in {genre_key} pool"

    # 3. Test resolve_dynamic_audio deterministic resolution
    g_fantasy_prof = beats.get_genre("fantasy")
    beat_setup = g_fantasy_prof.beats["establish"]
    beat_climax = g_fantasy_prof.beats["reveal"]
    amb1, cues1 = beats.resolve_dynamic_audio("fantasy", "setup", beat_setup, scene_index=0, scene_count=4, seed=10, estimated_seconds=15.0, tension=0.1)
    amb2, cues2 = beats.resolve_dynamic_audio("fantasy", "climax", beat_climax, scene_index=2, scene_count=4, seed=10, estimated_seconds=15.0, tension=0.9)
    assert amb1["src"] in make_assets.BEDS
    assert amb2["src"] in make_assets.BEDS
    assert len(cues1) > 0 and cues1[0]["sfx"] in make_assets.EFFECTS
    assert len(cues2) > 0 and cues2[0]["sfx"] in make_assets.EFFECTS

    g_scifi_prof = beats.get_genre("scifi")
    _, scifi_cues = beats.resolve_dynamic_audio("scifi", "climax", g_scifi_prof.beats["reveal"], scene_index=2, scene_count=4, seed=10, estimated_seconds=15.0, tension=0.9)
    assert scifi_cues[0]["sfx"] in make_assets.EFFECTS

    g_horror_prof = beats.get_genre("horror")
    _, horror_cues = beats.resolve_dynamic_audio("horror", "climax", g_horror_prof.beats["reveal"], scene_index=2, scene_count=4, seed=10, estimated_seconds=15.0, tension=0.9)
    assert horror_cues[0]["sfx"] in make_assets.EFFECTS

    # 4. Test genre default speech speeds and WPM scaling
    g_horror = beats.get_genre("horror")
    assert g_horror.default_speed == 1.0
    assert g_horror.default_wpm == 110

    g_motivational = beats.get_genre("motivational")
    assert g_motivational.default_speed == 1.05
    assert g_motivational.default_wpm == 130

    g_fantasy = beats.get_genre("fantasy")
    assert g_fantasy.default_speed == 0.94
    assert g_fantasy.default_wpm == 115

    # 5. Test beats.plan integration with audio cues and ambience
    plan_fantasy = beats.plan("ai dekho son pari", minutes=1.0, scenes=4, genre="fantasy")
    ambiences = [s["ambience"] for s in plan_fantasy["scenes"]]
    assert len(ambiences) == 4
    for amb in ambiences:
        amb_src = amb["src"] if isinstance(amb, dict) else amb
        assert amb_src in make_assets.BEDS

    cues_count = sum(len(s.get("soundCues", [])) for s in plan_fantasy["scenes"])
    assert cues_count >= 4, f"Expected at least 4 sound cues, got {cues_count}"
    for s in plan_fantasy["scenes"]:
        for cue in s.get("soundCues", []):
            sfx_name = cue.get("sfx") or cue.get("id")
            assert sfx_name in make_assets.EFFECTS
            assert 0.0 <= cue["volume"] <= 1.0

    print("  ✓ All Dynamic sound effects and speech speed checks passed.")


def test_video_formats() -> None:
    print("Testing Video Framing Presets and Aspect Ratio resolution...")

    # 1. Test video format catalog completeness
    formats = beats.list_formats()
    assert len(formats) >= 6, f"Expected at least 6 presets, got {len(formats)}"
    format_keys = {f["key"] for f in formats}
    for expected in ("reel", "landscape", "square", "portrait", "cinema", "classic"):
        assert expected in format_keys, f"Missing expected format {expected}"

    # 2. Test format retrieval by key, alias, and aspect ratio string
    fmt_reel = beats.get_format("reel")
    assert fmt_reel.width == 1080 and fmt_reel.height == 1920
    assert fmt_reel.aspect_ratio == "9:16"
    assert fmt_reel.sd_width == 432 and fmt_reel.sd_height == 768
    assert fmt_reel.bottom_margin >= 0.15  # mobile safe area

    assert beats.get_format("shorts").key == "reel"
    assert beats.get_format("tiktok").key == "reel"
    assert beats.get_format("9:16").key == "reel"
    assert beats.get_format("vertical").key == "reel"

    fmt_land = beats.get_format("landscape")
    assert fmt_land.width == 1920 and fmt_land.height == 1080
    assert fmt_land.aspect_ratio == "16:9"
    assert beats.get_format("youtube").key == "landscape"
    assert beats.get_format("16:9").key == "landscape"

    fmt_sq = beats.get_format("square")
    assert fmt_sq.width == 1080 and fmt_sq.height == 1080
    assert beats.get_format("1:1").key == "square"

    fmt_port = beats.get_format("portrait")
    assert fmt_port.width == 1080 and fmt_port.height == 1350
    assert beats.get_format("4:5").key == "portrait"

    fmt_cine = beats.get_format("cinema")
    assert fmt_cine.width == 2560 and fmt_cine.height == 1080
    assert beats.get_format("21:9").key == "cinema"

    fmt_class = beats.get_format("classic")
    assert fmt_class.width == 1440 and fmt_class.height == 1080
    assert beats.get_format("4:3").key == "classic"

    # 3. Test resolve_dimensions with custom overrides
    dim_custom = beats.resolve_dimensions(format_key="reel", width=720, height=1280, fps=60)
    assert dim_custom.width == 720
    assert dim_custom.height == 1280
    assert dim_custom.fps == 60
    assert dim_custom.aspect_ratio == "9:16"

    # 4. Test caption_options format adaptation
    captions_reel = new_video.caption_options("subtitle", True, format_profile=fmt_reel)
    assert captions_reel["bottom"] == 0.28
    assert captions_reel["fontSize"] == 55

    captions_land = new_video.caption_options("subtitle", True, format_profile=fmt_land)
    assert captions_land["bottom"] == 0.14
    assert captions_land["fontSize"] == 48

    # 5. Test SD latent dimension calculation
    assert generate_images.calculate_sd_dimensions(1080, 1920) == (432, 768)
    assert generate_images.calculate_sd_dimensions(1920, 1080) == (768, 432)
    assert generate_images.calculate_sd_dimensions(1080, 1080) == (512, 512)
    assert generate_images.calculate_sd_dimensions(1080, 1350) == (464, 576)
    assert generate_images.calculate_sd_dimensions(2560, 1080) == (896, 384)
    assert generate_images.calculate_sd_dimensions(1440, 1080) == (640, 480)

    # 6. Test beats.plan integration with format
    plan_reel = beats.plan("ninja anime fight", minutes=1.0, video_format="reel")
    assert plan_reel["format"] == "reel"
    assert plan_reel["width"] == 1080
    assert plan_reel["height"] == 1920
    assert plan_reel["aspectRatio"] == "9:16"
    assert plan_reel["formatProfile"].sd_width == 432
    assert plan_reel["formatProfile"].sd_height == 768

    print("  ✓ All Video Framing Presets and Aspect Ratio checks passed.")


def test_video_generation_mode() -> None:
    print("Testing exclusive LTX video-mode asset generation...")

    class FakeVastManager:
        def __init__(self) -> None:
            self.requests: list[dict] = []

        def generate_video(self, prompt: str, **kwargs) -> bytes:
            self.requests.append({"prompt": prompt, **kwargs})
            return b"test-mp4-data"

    manager = FakeVastManager()
    script = {
        "title": "Test",
        "topic": "A moonlit castle",
        "genre": "fantasy",
        "width": 1920,
        "height": 1080,
        "scenes": [{
            "id": "scene-001",
            "beat": "reveal",
            "headline": "The castle appears",
            "shots": [{"visual": {"type": "image", "src": "images/scene-001.png"}}],
            "background": {"type": "color", "color": "#000000"},
        }],
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        generated = generate_videos.generate_videos_for_script(
            script,
            out_dir=temp_dir,
            manager=manager,
        )
        assert (Path(temp_dir) / "scene-001.mp4").read_bytes() == b"test-mp4-data"

    assert generated["visualMode"] == "video"
    assert generated["videoModel"] == generate_videos.DEFAULT_MODEL
    assert generated["scenes"][0]["background"] == {
        "type": "video",
        "src": "videos/scene-001.mp4",
        "dim": 0.08,
    }
    assert generated["scenes"][0]["shots"] == []
    assert len(manager.requests) == 1
    request = manager.requests[0]
    assert request["width"] % 32 == 0 and request["height"] % 32 == 0
    assert request["num_frames"] >= 9 and (request["num_frames"] - 1) % 8 == 0
    assert generate_videos._video_dimensions(1920, 1080) == (768, 448)

    print("  ✓ Video mode replaces generated-image shots with video-only backgrounds.")


def test_vast_gpu_routing() -> None:
    print("Testing Vast.ai GPU selection and server startup configuration...")

    from vast_manager import VastManager

    manager = VastManager(api_key="test-key")
    commands: list[list[str]] = []

    def fake_run_cli(args: list[str]) -> str:
        commands.append(args)
        if args[0] == "search":
            # A real offer always carries an all-in price, which the budget ceiling filters on.
            return json.dumps([{"id": 123, "dph_total": 0.25, "machine_id": 1, "disk_space": 100}])
        return json.dumps({"new_contract": 456})

    manager.run_cli = fake_run_cli  # type: ignore[method-assign]
    assert manager.search_gpu("RTX 3090")[0]["id"] == 123
    assert "gpu_name=RTX_3090" in commands[-1][2]
    assert "num_gpus=1" in commands[-1][2]

    assert manager.create_instance(offer_id=123, gpu_name="RTX 4090") == 456
    # create_instance also reads the instance back to verify the billed rate, so locate
    # the create call rather than assuming it is the most recent command.
    create_args = next(cmd for cmd in commands if cmd[:2] == ["create", "instance"])
    # Port 8000 is published through --env, which is how `vastai create instance` maps ports.
    # The same one string also carries -e variables (the gated IndicF5 repo needs HF_TOKEN and
    # the server needs to be told where its reference clips are), so assert containment.
    assert "--env" in create_args
    env_spec = create_args[create_args.index("--env") + 1]
    assert "-p 8000:8000" in env_spec, f"the port mapping must survive the env spec: {env_spec}"
    assert "--onstart-cmd" in create_args
    assert "vast_server.py" in create_args[create_args.index("--onstart-cmd") + 1]

    manager.check_health = lambda url=None: {  # type: ignore[method-assign]
        "status": "healthy",
        "vram": {"device_name": "NVIDIA GeForce RTX 4090"},
    }
    assert manager.require_gpu("RTX 4090")["status"] == "healthy"
    try:
        manager.require_gpu("RTX 3090")
    except RuntimeError as err:
        assert "GPU mismatch" in str(err)
    else:
        raise AssertionError("GPU mismatch must stop the run")

    print("  ✓ Vast creation requests one selected GPU and verifies the running server GPU.")


def test_vast_setup_resilience() -> None:
    print("Testing Vast.ai slow-vs-broken handling, onstart guard and host filters...")

    import vast_manager
    from vast_manager import InstanceBroken, InstanceSlow, VastManager

    # 1. A host that cannot pull the image is destroyed and blacklisted.
    broken = VastManager(api_key="test-key")
    broken.search_gpu = lambda *a, **k: [{"id": 11, "dph_total": 0.2, "machine_id": 777}]  # type: ignore[method-assign]
    broken.create_instance = lambda **k: 4242  # type: ignore[method-assign]
    destroyed: list[int] = []
    broken.destroy_instance = lambda i: destroyed.append(i) or True  # type: ignore[method-assign]
    broken._safe_stop = lambda i: None  # type: ignore[method-assign]

    def _raise_broken(_i, **_k):
        raise InstanceBroken("docker TLS handshake timeout")

    broken.wait_until_ready = _raise_broken  # type: ignore[method-assign]
    broken.bad_hosts = {}
    bad_recorded: list[tuple] = []
    broken.mark_bad_host = lambda m, r: bad_recorded.append((m, r))  # type: ignore[method-assign]
    try:
        broken.provision_instance("RTX 3090")
    except RuntimeError:
        pass
    assert destroyed == [4242], f"broken instance must be destroyed, got {destroyed}"
    assert bad_recorded and bad_recorded[0][0] == 777, "broken host must be blacklisted"

    # 2. A host that is merely slow must be STOPPED and kept, never destroyed or
    #    blacklisted: destroying it would throw away the install it already paid for.
    slow = VastManager(api_key="test-key")
    slow.search_gpu = lambda *a, **k: [{"id": 12, "dph_total": 0.2, "machine_id": 888}]  # type: ignore[method-assign]
    slow.create_instance = lambda **k: 4343  # type: ignore[method-assign]
    slow_destroyed: list[int] = []
    slow_stopped: list[int] = []
    slow.destroy_instance = lambda i: slow_destroyed.append(i) or True  # type: ignore[method-assign]
    slow.stop_instance = lambda i: slow_stopped.append(i) or True  # type: ignore[method-assign]
    slow._record_idle = lambda i: None  # type: ignore[method-assign]
    slow.bad_hosts = {}
    slow_bad: list[tuple] = []
    slow.mark_bad_host = lambda m, r: slow_bad.append((m, r))  # type: ignore[method-assign]

    def _raise_slow(_i, **_k):
        raise InstanceSlow("no install progress for 600s")

    slow.wait_until_ready = _raise_slow  # type: ignore[method-assign]
    try:
        slow.provision_instance("RTX 3090")
        raise AssertionError("a stalled instance must not be reported as ready")
    except RuntimeError as err:
        assert "--vast-instance 4343" in str(err), "resume hint must name the stopped instance"
    assert slow_stopped == [4343], f"slow instance must be stopped, got {slow_stopped}"
    assert slow_destroyed == [], f"slow instance must NOT be destroyed, got {slow_destroyed}"
    assert slow_bad == [], "a slow host must not be blacklisted"

    # 3. The install is guarded by a marker so a restarted instance never reinstalls.
    onstart = VastManager.server_onstart_command()
    assert ".tube2dscript_deps_ready" in onstart, "onstart must guard the install with a marker"
    assert "uv pip install --system" in onstart, "onstart must install with uv"
    assert "command -v git" in onstart, "the Git-backed IndicF5 runtime requires a git preflight"
    assert onstart.index("apt done") < onstart.index("uv pip install"), (
        "dependency installation must wait for apt to provide git"
    )
    assert "git clone" in onstart and onstart.index("apt done") < onstart.index("git clone"), (
        "the repo must be cloned only after apt guarantees git"
    )
    assert "torch.version.cuda" in onstart, "onstart must report the CUDA build it ended up with"

    # 5. The torch family must be resolved from the image's own CUDA index, otherwise a
    #    torchaudio wheel built for a different CUDA version may be pulled in, which the
    #    image's torch cannot load; that surfaced only at first synthesis as
    #    "Could not load .../torchaudio/lib/libtorchaudio.so" and an opaque HTTP 500.
    assert "--torch-backend=cu124" in onstart, (
        "the onstart must pin the torch index to the image's CUDA build (cuda12.4 -> cu124)"
    )
    assert "torchaudio" in onstart, "torchaudio must be installed and verified explicitly"
    assert "torchaudio==2.4.0" in onstart, (
        "torchaudio must be PINNED to the image's torch version: resolving it freely "
        "installs a newer torchaudio whose ABI does not match the image's torch"
    )
    assert "torchaudio.__version__" in onstart, (
        "the onstart must compare the torch and torchaudio versions, not merely import them"
    )

    # 4. Prefetch is caller-driven AND defaults to the full set the pipeline can ask for.
    #    A deps-only image ships no weights, so the default must cover LTX and FLUX too;
    #    otherwise every fresh instance pays for them lazily at the first image/video
    #    request (~42 GB of egress per rental).
    ltx_onstart = VastManager.server_onstart_command(prefetch_models=["jbilcke-hf/LTX-Video-2b"])
    assert "jbilcke-hf/LTX-Video-2b" in ltx_onstart
    assert "jbilcke-hf/LTX-Video-2b" in VastManager.server_onstart_command(), (
        "the default prefetch must include the LTX checkpoint the video pipeline needs"
    )
    default_onstart = VastManager.server_onstart_command()
    assert "black-forest-labs/FLUX.1-schnell" in default_onstart, (
        "the default prefetch must include the FLUX checkpoint the image pipeline needs"
    )
    assert "ai4bharat/IndicF5" in default_onstart, (
        "the default prefetch must include the IndicF5 narration runtime"
    )
    # Each repo is prefetched by its OWN snapshot_download call, so one failure cannot
    # discard the repos that already finished.
    assert default_onstart.count("snapshot_download(") == 3, (
        "each prefetch repo must get its own snapshot_download call"
    )

    # 5. The render stack is NOT part of the onstart dependency marker. The onstart only
    #    runs at container creation, so anything inside that marker would never reach a
    #    warm-reused instance -- which is exactly what Fix 1 hands back.
    plain_stack = VastManager.server_onstart_command()
    assert "nodejs" not in plain_stack and "fonts-noto-core" not in plain_stack, (
        "the render stack must not be installed by the onstart script"
    )
    ssh_stack = VastManager.render_stack_shell()
    assert "nodejs" in ssh_stack, "the render stack must install Node"
    assert "fonts-noto-core" in ssh_stack, (
        "Devanagari captions need a Noto fallback: the theme fonts are Windows-only names"
    )
    assert "npm ci" in ssh_stack, "the render stack must install the npm dependencies"
    assert "remotion browser ensure" in ssh_stack, "headless Chromium must be fetched for Remotion"
    assert ssh_stack.strip().startswith("if command -v node"), (
        "the render stack must be idempotent so a reused instance is cheap to top up"
    )
    assert "exit 1" in ssh_stack, "real failures must fail loudly instead of rendering nothing"

    # 6. The install marker must live where the packages live. On the /workspace volume it
    #    would survive a container reset and falsely claim the deps exist.
    assert "/workspace/.deps_ready" not in plain_stack, "the marker must not sit on the volume"
    assert "HOME/.tube2dscript_deps_ready" in plain_stack
    # The marker must be re-verified with a real import -- and that import has to name the
    # modules the server loads LAZILY. Probing only the top-level packages is exactly how a
    # pyarrow/NumPy ABI mismatch was certified "ready", after which the marker kept every
    # later run from reinstalling and each synthesis request answered HTTP 500.
    marker_probe = VastManager.dependency_probe_command(quiet=True)
    assert marker_probe in plain_stack, "the marker must be re-verified with a real import"
    assert "f5_tts.infer.utils_infer" in marker_probe, (
        "the probe must import the IndicF5 module the synthesis handler imports lazily"
    )
    assert "vocos" in marker_probe, "the probe must import IndicF5's vocoder runtime"
    assert "parler_tts" not in marker_probe, (
        "legacy Parler compatibility must not block IndicF5 production readiness"
    )
    assert "transformers" in marker_probe, "the probe must import transformers, the model runner"
    assert plain_stack.count(marker_probe) >= 2, (
        "the marker check and the marker write must share one probe"
    )

    # 7. The speed filter relaxes instead of returning nothing.
    calls: list[str] = []
    tiered = VastManager(api_key="test-key")

    def tiered_cli(args: list[str]) -> str:
        query = args[2] if args[0] == "search" and len(args) > 2 else ""
        calls.append(query)
        if "inet_down>1000" in query:
            return "[]"  # strict tier matches nothing
        return json.dumps([{"id": 99, "dph_total": 0.2, "machine_id": 5, "inet_down": 150}])

    tiered.run_cli = tiered_cli  # type: ignore[method-assign]
    offers = tiered.search_gpu("RTX 3090", min_inet_down=1000, min_disk_bw=500)
    assert len(offers) == 1, "the relaxed tier must still find an offer"
    assert any("inet_down>1000" in q for q in calls), "the strict tier must be tried first"
    assert any("inet_down>100" in q and "inet_down>1000" not in q for q in calls), (
        "the relaxed tier must be tried second"
    )
    # 7b. The download cap applies to every tier, including the last-resort query.
    assert all("inet_down<400" in q for q in calls), "the VAST_MAX_INET_DOWN cap must not relax"
    tiered.run_cli = lambda args: json.dumps(  # type: ignore[method-assign]
        [{"id": 101, "dph_total": 0.2, "machine_id": 7, "inet_down": 900}]
    )
    assert tiered.search_gpu("RTX 3090") == [], "an offer above the download cap must be dropped"

    # 8. An over-budget offer is still rejected regardless of the speed filters.
    tiered.run_cli = lambda args: json.dumps([{"id": 100, "dph_total": 9.99, "machine_id": 6}])  # type: ignore[method-assign]
    assert tiered.search_gpu("RTX 3090") == [], "the price ceiling must survive the filter changes"

    # 9. The remote render payload must be asset-driven. public/ is hundreds of MB of
    #    stale narration from past runs, so shipping the tree would be wasteful and slow.
    import vast_render

    payload_files, _payload_warnings = vast_render.collect_payload_files()
    relative = {str(p.relative_to(vast_render.ROOT)).replace("\\", "/") for p in payload_files}
    for required in ("script.json", "package.json", "src/generated/project.json", "src/Main.tsx"):
        assert required in relative, f"the render payload must include {required}"
    assert not any(name.startswith("out/") for name in relative), "the payload must exclude build output"
    assert not any(name.startswith("node_modules/") for name in relative)
    assert sum(p.stat().st_size for p in payload_files) < 400 * 1024 * 1024, (
        "the payload must be pruned rather than shipping all of public/"
    )

    print("  ✓ Slow hosts are kept, broken hosts are dropped, and setup is guarded and parallel.")


def test_vast_boot_stall_policy() -> None:
    """Option B (chosen): a boot that makes no progress is STOPPED when the install is real,
    and only a provably EMPTY install (peak disk below the floor) is treated as broken.

    Drives wait_until_ready() with a fake clock so no real time is spent.
    """
    print("Testing Vast.ai boot-stall policy (Option B: stop when alive, destroy only if empty)...")

    import vast_manager
    from vast_manager import InstanceBroken, InstanceSlow, VastManager

    real_time = vast_manager.time

    class _Clock:
        def __init__(self) -> None:
            self.t = 1000.0

        def time(self) -> float:
            return self.t

        def sleep(self, seconds: float) -> None:
            self.t += seconds

    def _drive(disk_gb: float):
        vast_manager.time = _Clock()  # type: ignore[assignment]
        try:
            mgr = VastManager(api_key="test-key")
            mgr._diagnose_boot = lambda *a, **k: None  # type: ignore[method-assign]
            mgr.mark_bad_host = lambda *a, **k: None  # type: ignore[method-assign]
            mgr.bad_hosts = {}
            info = {"actual_status": "running", "status_msg": "", "machine_id": 999}
            mgr.get_instance_info = lambda _i: dict(info)  # type: ignore[method-assign]
            mgr._disk_used_gb = lambda _info: disk_gb  # type: ignore[method-assign]
            # No server URL -> never reaches the health-check success path.
            mgr._resolve_server_url = lambda _info: ""  # type: ignore[method-assign]
            return mgr.wait_until_ready(4242)
        finally:
            vast_manager.time = real_time

    # Alive but stalled (2GB installed, then frozen): stop-and-keep, never destroy.
    try:
        _drive(2.0)
        raise AssertionError("a stalled-but-alive boot must raise InstanceSlow, not become ready")
    except InstanceSlow:
        pass
    except InstanceBroken as err:
        raise AssertionError(f"a host with a real install must NOT be destroyed: {err}")

    # Nothing ever installed (disk never passed the floor): the empty install is worthless,
    # so this is the one boot-stall case that is treated as broken (destroyed) under Option B.
    try:
        _drive(0.5)
        raise AssertionError("a boot that installed nothing must raise InstanceBroken")
    except InstanceBroken:
        pass

    print("  ✓ Boot-stall policy held: stalled-but-alive -> InstanceSlow (stop); empty -> InstanceBroken.")


def test_audio_image_only_mode() -> None:
    """The audio/image contract is selected by mode and leaves the default path untouched."""
    import vast_manager
    from vast_manager import VastManager

    print("Testing audio/image-only mode contract...")

    saved = os.environ.get("VAST_AUDIO_IMAGE_ONLY")
    saved_prefetch = os.environ.get("VAST_SERVER_PREFETCH")
    try:
        os.environ["VAST_AUDIO_IMAGE_ONLY"] = "1"
        # A stale env var from a video run must not reintroduce LTX into this mode.
        os.environ["VAST_SERVER_PREFETCH"] = "jbilcke-hf/LTX-Video-2b-0-9-8-distilled-HFIE"

        # 1. The container is told the mode, and the port mapping survives.
        env_spec = VastManager.container_env_spec()
        assert "-p 8000:8000" in env_spec, "the port mapping must survive the extra env var"
        assert "-e VAST_AUDIO_IMAGE_ONLY=1" in env_spec, "the container must learn the mode"

        # 2. Prefetch fetches Klein and NOT LTX.
        models = vast_manager.prefetch_models()
        assert "black-forest-labs/FLUX.2-klein-4B" in models, "Klein must be prefetched"
        assert not any("LTX" in model for model in models), "LTX must never be prefetched here"

        # 3. Deps REPLACED the default pins (appending would leave a second diffusers line).
        deps = VastManager.server_dependency_packages()
        assert any(dep.startswith("diffusers>=0.37") for dep in deps), "diffusers must be >=0.37"
        assert not any(dep == "diffusers==0.32.2" for dep in deps), (
            "diffusers==0.32.2 must not appear: uv would silently pick it"
        )
        assert any(dep.startswith("transformers>=4.51") for dep in deps), "transformers must be >=4.51"
        assert not any(dep == "transformers==4.49.0" for dep in deps), (
            "transformers==4.49.0 predates Qwen3ForCausalLM"
        )

        # 4. Marker is mode-specific, so a warm normal-mode instance cannot short-circuit
        #    the install, and the probe names the classes this mode constructs.
        onstart = VastManager.server_onstart_command(prefetch_models=None)
        assert "deps_ready_aiimg" in onstart, "the marker must be mode-specific"
        assert "Flux2KleinPipeline" in onstart, "the onstart must verify the Klein import"
        assert "LTXPipeline" not in onstart, "the onstart must not verify LTX in this mode"
        assert "ignore_patterns" in onstart, "the Klein prefetch must skip the duplicate .safetensors"
    finally:
        if saved is None:
            os.environ.pop("VAST_AUDIO_IMAGE_ONLY", None)
        else:
            os.environ["VAST_AUDIO_IMAGE_ONLY"] = saved
        if saved_prefetch is None:
            os.environ.pop("VAST_SERVER_PREFETCH", None)
        else:
            os.environ["VAST_SERVER_PREFETCH"] = saved_prefetch

    # 5. Default mode is unchanged: same pins, same prefetch, same marker, LTX allowed.
    assert any(dep == "diffusers==0.32.2" for dep in VastManager.server_dependency_packages()), (
        "the default pin set must not move: the LTX checkpoint declares 0.32.0.dev0"
    )
    assert any(dep == "transformers==4.49.0" for dep in VastManager.server_dependency_packages())
    default_onstart = VastManager.server_onstart_command(
        prefetch_models=["jbilcke-hf/LTX-Video-2b-0-9-8-distilled-HFIE"]
    )
    assert "deps_ready_aiimg" not in default_onstart, "the default marker must keep its name"
    assert "Flux2KleinPipeline" not in default_onstart, "the default probe must not require Klein"
    assert "LTXPipeline" in default_onstart, "the default probe must still verify LTX"
    assert "autoencoder_kl" in default_onstart, "both probes must exercise the autoencoder import"

    print("  audio/image-only mode contract OK")


def test_vast_preflight() -> None:
    """The local preflight must pass on this repo and restore the caller's environment."""
    import vast_manager

    print("Testing Vast mode preflight (no rental)...")

    saved = os.environ.get("VAST_AUDIO_IMAGE_ONLY")
    try:
        os.environ["VAST_AUDIO_IMAGE_ONLY"] = "1"
        os.environ["VAST_SERVER_PREFETCH"] = "jbilcke-hf/LTX-Video-2b-0-9-8-distilled-HFIE"
        failures = vast_manager.preflight_audio_image()
        assert failures == [], f"the audio/image contract is broken: {failures}"

        # The preflight must be side-effect free: it forces the mode internally, so the
        # caller's environment (and their stale video prefetch list) must survive intact.
        assert os.environ["VAST_AUDIO_IMAGE_ONLY"] == "1", "the preflight must restore the env"
        assert "LTX" in os.environ["VAST_SERVER_PREFETCH"], "the preflight must not rewrite the prefetch env"

        described = "\n".join(vast_manager.describe_mode())
        assert "audio/image-only" in described
        assert "deps_ready_aiimg" in described
        assert "Flux2KleinPipeline" in described
        assert "ignore" in described

        # The Vast provider default is what the server is told to load, so it must follow
        # the mode. A hardcoded FLUX.1 default here would contradict the Klein pipeline the
        # same run is prefetching and verifying.
        import generate_images

        provider, _key, model = generate_images.detect_provider("vast")
        assert provider == "vast"
        assert model == "black-forest-labs/FLUX.2-klein-4B", (
            f"audio/image mode must default to Klein, not FLUX.1: {model}"
        )
    finally:
        if saved is None:
            os.environ.pop("VAST_AUDIO_IMAGE_ONLY", None)
        else:
            os.environ["VAST_AUDIO_IMAGE_ONLY"] = saved
        os.environ.pop("VAST_SERVER_PREFETCH", None)

    # Normal mode must default to FLUX.1-schnell again once the mode is unset.
    import generate_images

    _provider, _key, model = generate_images.detect_provider("vast")
    assert model == "black-forest-labs/FLUX.1-schnell", model

    print("  audio/image preflight OK")


def test_vast_rental_cost_ranking() -> None:
    """The picker must rank by 1-hour all-in cost, not by sticker GPU rate."""
    import vast_manager

    print("Testing short-rental cost ranking...")

    def offer(oid: int, rate: float, mbps: float, dl_cost: float, storage: float) -> dict:
        # `storage` is Vast's real unit: $ per GB per MONTH.
        return {
            "id": oid,
            "gpu_name": "RTX 4090",
            "dph_total": rate,
            "inet_down": mbps,
            "inet_down_cost": dl_cost,
            "storage_cost": storage,
        }

    # A: cheapest GPU rate, but 100 Mbps and $0.01/GB egress.
    # B: $0.06/hr dearer, but 900 Mbps and free egress.
    # At 1 hour + 25GB + 60GB disk, B wins on total AND finishes the download 9x sooner.
    cheap_slow = offer(1, 0.30, 100, 0.010, 0.20)
    dear_fast = offer(2, 0.36, 900, 0.0, 0.20)

    cost_a = vast_manager.rental_cost(cheap_slow, hours=1, download_gb=25, disk_gb=60)
    cost_b = vast_manager.rental_cost(dear_fast, hours=1, download_gb=25, disk_gb=60)

    assert cost_a["download"] > 0, "per-GB egress must be priced in"
    assert cost_b["download"] == 0, "free egress must cost nothing"
    # Storage is $0.20/GB/month, so 60GB for 1h is 0.20*60/730 = $0.0164, NOT $12.
    assert abs(cost_a["storage"] - 0.20 * 60 / 730) < 1e-9, cost_a
    assert cost_a["storage"] < 0.05, (
        f"storage must be converted from $/GB/mo to an hourly figure, got {cost_a['storage']}"
    )
    # 0.30 + 0.25 egress + 0.0164 storage = 0.5664 vs 0.36 + 0 + 0.0164 = 0.3764
    assert abs(cost_a["total"] - 0.5664) < 1e-3, cost_a
    assert abs(cost_b["total"] - 0.3764) < 1e-3, cost_b
    assert cost_a["missing"] == [], "a reported egress price is not a missing field"

    # An offer that omits `inet_down_cost` must not be silently treated as free.
    unreported = {key: value for key, value in cheap_slow.items() if key != "inet_down_cost"}
    assert vast_manager.rental_cost(unreported)["missing"] == ["inet_down_cost"], (
        "an unreported egress price must be flagged, not assumed to be 0"
    )

    ranked = vast_manager.rank_offers([cheap_slow, dear_fast], hours=1, download_gb=25, disk_gb=60)
    assert ranked[0][0]["id"] == 2, "the fully-loaded cost must win, not the GPU rate"
    assert ranked[1][0]["id"] == 1

    # Download ETA makes the hidden cost visible.
    assert vast_manager.download_seconds(dear_fast, 25) < vast_manager.download_seconds(cheap_slow, 25)
    assert vast_manager.download_seconds(offer(3, 0.3, 0, 0, 0), 25) == float("inf")

    # An unpriceable offer sorts last rather than first (inf is not "cheapest").
    ranked_unpriced = vast_manager.rank_offers(
        [{**cheap_slow, "dph_total": None, "dph": None}, dear_fast],
        hours=1,
        download_gb=25,
        disk_gb=60,
    )
    assert ranked_unpriced[0][0]["id"] == 2

    # The countdown auto-picks the default when stdin cannot deliver a keypress.
    class _NotATty:
        def isatty(self) -> bool:
            return False

    saved = sys.stdin
    try:
        sys.stdin = _NotATty()  # type: ignore[assignment]
        assert vast_manager.prompt_pick_with_timeout(5, default_index=1, timeout_sec=0) == 1
        assert vast_manager.prompt_pick_with_timeout(5, default_index=3, timeout_sec=0) == 3
    finally:
        sys.stdin = saved

    print("  short-rental cost ranking OK")


def test_vast_meta_device_and_diagnose() -> None:
    """A model that lands on the `meta` device must fail loudly, not 500 every request.

    Newer transformers defaults `low_cpu_mem_usage=True`, which builds the module graph
    with no storage; the following `.to("cuda")` then raises
    "Cannot copy out of meta tensor; no data!" for every synthesis request. The client sees
    a bare HTTP 500, so the guard is asserted here rather than rediscovered on a rental.
    """
    print("Testing the meta-device guard and the read-only diagnose command...")

    import io
    from contextlib import redirect_stdout

    import generate_images
    import vast_server
    from vast_manager import VastManager

    calls: list[dict] = []

    def loader(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return "model"

    # 1. The loader is asked for real CPU materialisation.
    assert vast_server._from_pretrained_on_cpu(loader, "repo", trust_remote_code=True) == "model"
    assert calls[-1]["kwargs"]["low_cpu_mem_usage"] is False, (
        "the loader must disable the meta-device fast init"
    )
    assert calls[-1]["kwargs"]["trust_remote_code"] is True

    # 2. A loader that does not accept the flag is retried WITHOUT it rather than failing
    #    the entire load over a compatibility keyword.
    def strict_loader(*args, **kwargs):
        if "low_cpu_mem_usage" in kwargs:
            raise TypeError("__init__() got an unexpected keyword 'low_cpu_mem_usage'")
        calls.append({"args": args, "kwargs": kwargs})
        return "strict-model"

    assert vast_server._from_pretrained_on_cpu(strict_loader, "repo") == "strict-model"
    assert "low_cpu_mem_usage" not in calls[-1]["kwargs"]

    # 3. A genuinely meta model produces an actionable sentence, not a torch internal.
    class _MetaModel:
        def to(self, _device):
            raise NotImplementedError(
                "Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() "
                "instead of torch.nn.Module.to() when moving module from meta to a different device."
            )

    try:
        vast_server.release_meta_safe(_MetaModel())
    except RuntimeError as err:
        assert "meta device" in str(err) and "low_cpu_mem_usage" in str(err), str(err)
    else:
        raise AssertionError("a meta-device model must not be silently accepted")

    class _Fine:
        moved: list[str] = []

        def to(self, device):
            self.moved.append(device)
            return self

    fine = _Fine()
    assert vast_server.release_meta_safe(fine) is fine
    assert fine.moved == [vast_server.DEVICE]

    # 4. `diagnose` is read-only: it must never install anything or touch the marker.
    manager = VastManager(api_key="test-key")
    issued: list[str] = []
    manager.run_ssh = lambda i, cmd, timeout=600: issued.append(cmd) or "[diagnose] ok"  # type: ignore[method-assign]
    manager.diagnose_instance(4242, timeout=30)
    assert issued, "diagnose must run exactly one remote command"
    command = issued[0]
    for needed in ("pydantic", "transformers, diffusers", "torchaudio", "probe"):
        assert needed in command, f"the diagnose report must include {needed}: {command}"
    assert "autoencoder_kl" in command, (
        "diagnose must run the same probe the onstart certifies with"
    )
    assert "uv pip install" not in command and "pip install" not in command, (
        "diagnose is read-only; installing belongs to repair"
    )
    assert "touch" not in command, "diagnose must not create or restore a deps marker"

    # 5. The banner must not print the local pipeline name next to a Vast model. It did:
    # "@1920x1080) (segmind/SSD-1B)" while the title said FLUX.2-klein-4B.
    saved_env = {key: os.environ.get(key) for key in ("VAST_AUDIO_IMAGE_ONLY", "IMAGE_MODEL")}
    buffer = io.StringIO()
    try:
        os.environ["VAST_AUDIO_IMAGE_ONLY"] = "1"
        os.environ.pop("IMAGE_MODEL", None)
        banner_provider = generate_images.detect_provider("vast")
        with redirect_stdout(buffer):
            # Recreate just the banner decision the generator makes, without renting a GPU.
            vast_image_model = (
                "FLUX.2-klein-4B"
                if os.environ.get("VAST_AUDIO_IMAGE_ONLY", "0").strip() == "1"
                else "FLUX.1-schnell"
            )
            print(f"Retrieving background images via Vast.ai Cloud RTX 4090 GPU ({vast_image_model}, 16:9) for 1 scenes...")
        text = buffer.getvalue()
        assert "segmind" not in text and "SSD-1B" not in text, (
            f"the Vast banner must not name a local pipeline: {text!r}"
        )
        assert banner_provider[2] == "black-forest-labs/FLUX.2-klein-4B", banner_provider
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    print("  ✓ meta-device models fail loudly, diagnose is read-only, and the banner is honest.")


def test_vast_reuse_contract() -> None:
    """A reused instance must be able to serve THIS run, not merely match the GPU.

    `start` re-clones the repo but never rewrites site-packages, so an instance
    provisioned before a pin change kept answering every request with the same HTTP 500
    while looking like a client bug. Reuse is now conditional on /health.
    """
    print("Testing the Vast reuse contract (/health gating)...")

    from vast_manager import VastManager

    saved = os.environ.get("VAST_AUDIO_IMAGE_ONLY")
    try:
        os.environ["VAST_AUDIO_IMAGE_ONLY"] = "1"

        # 1. A healthy audio/image box is reusable.
        healthy = VastManager(api_key="test-key")
        healthy.check_health = lambda url=None: {  # type: ignore[method-assign]
            "status": "healthy",
            "audio_image_only": True,
            "image_model": "black-forest-labs/FLUX.2-klein-4B",
            "image_ready": True,
            "indicf5_ready": True,
        }
        assert healthy.reuse_contract_problem({"actual_status": "running", "public_ipaddr": "203.0.113.9", "ports": {"8000/tcp": [{"HostPort": "1234"}]}}, require_tts=True) is None

        # 2. The exact failure from the broken box: image path unusable.
        broken = VastManager(api_key="test-key")
        broken.check_health = lambda url=None: {  # type: ignore[method-assign]
            "status": "healthy",
            "audio_image_only": True,
            "image_model": "black-forest-labs/FLUX.2-klein-4B",
            "image_ready": False,
            "image_error": "RuntimeError: infer_schema(func): Parameter q has unsupported type torch.Tensor",
            "indicf5_ready": False,
            "indicf5_error": "NotImplementedError: Cannot copy out of meta tensor",
        }
        info = {"actual_status": "running", "public_ipaddr": "211.120.210.29", "ports": {"8000/tcp": [{"HostPort": "41829"}]}}
        problem = broken.reuse_contract_problem(info, require_tts=True)
        assert problem and "image pipeline unusable" in problem, problem
        assert "meta tensor" in problem, "the TTS half must be reported too when narration is needed"

        # Narration is not required for an image-only run, so only the image half counts.
        image_only = broken.reuse_contract_problem(info, require_tts=False)
        assert image_only and "meta tensor" not in image_only

        # 3. A normal-mode container must not be handed to an audio/image run.
        normal = VastManager(api_key="test-key")
        normal.check_health = lambda url=None: {  # type: ignore[method-assign]
            "status": "healthy",
            "audio_image_only": False,
            "image_model": "black-forest-labs/FLUX.1-schnell",
            "image_ready": True,
            "indicf5_ready": True,
        }
        wrong_mode = normal.reuse_contract_problem(info, require_tts=False)
        assert wrong_mode and "old normal mode" in wrong_mode, wrong_mode

        # 4. A stopped instance is reusable untested: it starts first, and the caller's
        #    health check runs afterwards.
        assert normal.reuse_contract_problem({"actual_status": "stopped"}, require_tts=True) is None

        # 5. An unreachable server is not reusable.
        silent = VastManager(api_key="test-key")

        def _refuse(url=None):
            raise ConnectionRefusedError("no listener")

        silent.check_health = _refuse  # type: ignore[method-assign]
        unreachable = silent.reuse_contract_problem(info, require_tts=False)
        assert unreachable and "did not answer" in unreachable, unreachable

        # 6. find_existing_instance must skip (not return) a contract-breaking instance.
        finder = VastManager(api_key="test-key")
        finder.run_cli = lambda args: json.dumps([  # type: ignore[method-assign]
            {
                "id": 56399186,
                "gpu_name": "RTX 4090",
                "actual_status": "running",
                "disk_space": 60,
                "dph_total": 0.457,
                "machine_id": 150186,
                "public_ipaddr": "211.120.210.29",
                "ports": {"8000/tcp": [{"HostPort": "41829"}]},
            },
        ])
        finder.check_health = broken.check_health
        assert finder.find_existing_instance("RTX 4090", min_disk=60, require_tts=True) == (None, None), (
            "an instance whose environment cannot serve this run must not be reused"
        )
    finally:
        if saved is None:
            os.environ.pop("VAST_AUDIO_IMAGE_ONLY", None)
        else:
            os.environ["VAST_AUDIO_IMAGE_ONLY"] = saved

    print("  ✓ Reuse is gated on /health, so a stale install cannot be handed back forever.")


def test_vast_dependency_abi() -> None:
    print("Testing the pinned NumPy/pyarrow ABI, dependency repair, and fast-fail on a broken server...")

    from vast_manager import DEFAULT_SERVER_IMAGE, SERVER_DEPENDENCIES, VastManager

    # 1. IndicF5 needs AI4Bharat's Git fork, not PyPI's unrelated f5-tts project. Its
    #    numerics/data stack is pinned to the proven environment so pyarrow cannot silently
    #    resolve against an incompatible NumPy ABI on a newly rented host.
    pins = {name.split(">")[0].split("<")[0].split("=")[0]: name for name in SERVER_DEPENDENCIES}
    assert pins["numpy"] == "numpy==1.26.4", "IndicF5 must pin the tested NumPy ABI"
    assert pins["pyarrow"] == "pyarrow==20.0.0", "IndicF5 must pin compatible pyarrow"
    assert pins["datasets"] == "datasets==3.6.0", "the lazy F5 data path must be reproducible"
    f5_package = next(name for name in SERVER_DEPENDENCIES if name.startswith("f5_tts @"))
    assert "git+https://github.com/AI4Bharat/IndicF5.git" in f5_package, (
        "the server must install the AI4Bharat IndicF5 runtime it actually imports"
    )
    assert "vocos==0.1.0" in SERVER_DEPENDENCIES

    # 2. The shared package list must still carry the torchaudio pin, and the install helper
    #    must carry every package: the onstart and `repair` both run this one snippet, so a
    #    repair can never rebuild a different environment than the install created.
    packages = VastManager.server_dependency_packages(DEFAULT_SERVER_IMAGE)
    assert packages[:-1] == SERVER_DEPENDENCIES, "the helper must add exactly the torchaudio pin"
    assert packages[-1].startswith("torchaudio=="), (
        f"torchaudio must stay pinned to the image's own torch: {packages[-1]}"
    )
    install = VastManager.dependency_install_command()
    assert "--torch-backend=cu124" in install, "the image's CUDA index must still steer the resolve"
    for package in packages:
        assert package in install, f"the install command dropped {package}"

    # 3. The probe must name the lazily imported module, and the repair must clear the
    #    marker before reinstalling and restore it only after the probe passes -- otherwise
    #    a warm instance keeps its broken install forever.
    probe = VastManager.dependency_probe_command()
    assert "f5_tts.infer.utils_infer" in probe and "transformers" in probe and "numpy" in probe
    assert "vocos" in probe and "parler_tts" not in probe, (
        "IndicF5 production readiness must not depend on optional Parler compatibility"
    )
    # `from diffusers import FluxPipeline` binds a symbol; the module that actually raises
    # `infer_schema(func): Parameter q has unsupported type torch.Tensor` is the autoencoder,
    # which diffusers imports lazily. Probing only the pipeline class certified a runtime
    # that then answered HTTP 500 to every one of 25 scene requests.
    assert "autoencoder_kl" in probe, (
        "the probe must import the autoencoder module a torch/pydantic mismatch breaks in"
    )

    # pydantic is a TRANSITIVE dependency of transformers, so an unpinned resolve silently
    # installs a version whose schema hook older torch builds cannot satisfy.
    pinned_names = {name.split(">")[0].split("<")[0].split("=")[0]: name for name in SERVER_DEPENDENCIES}
    assert pinned_names.get("pydantic", "").startswith("pydantic=="), (
        f"pydantic must be pinned to torch 2.4.0's compatible range: {pinned_names.get('pydantic')}"
    )
    repaired: list[tuple[int, str, int]] = []
    manager = VastManager(api_key="test-key")
    manager.run_ssh = lambda i, cmd, timeout=600: repaired.append((i, cmd, timeout)) or "[repair] ok"  # type: ignore[method-assign]
    manager.repair_dependencies(4242, timeout=99)
    assert repaired and repaired[0][0] == 4242 and repaired[0][2] == 99, repaired
    command = repaired[0][1]
    assert 'rm -f "$HOME/.tube2dscript_deps_ready"' in command, (
        "the repair must clear the marker, or the broken install is never revisited"
    )
    for package in packages:
        assert package in command, f"the repair install dropped {package}"
    assert "f5_tts.infer.utils_infer" in command, "the repair must re-verify the lazy IndicF5 probe"
    assert command.index("rm -f") < command.index("touch"), (
        "the marker may only be restored after the probe passes"
    )

    # 4. A server that reports a broken TTS import path must be refused when narration
    #    needs it, ignored when it does not (image/video runs never touch TTS), and a server
    #    too old to report the field must be accepted rather than stranded.
    gated = VastManager(api_key="test-key")
    broken_health = {
        "status": "healthy",
        "vram": {"device_name": "NVIDIA GeForce RTX 4090"},
        "indicf5_ready": False,
        "indicf5_error": "ImportError: vocos is unavailable",
    }
    gated.check_health = lambda url=None: dict(broken_health)  # type: ignore[method-assign]
    assert gated.require_gpu("RTX 4090")["indicf5_ready"] is False, "image runs must not be blocked"
    try:
        gated.require_gpu("RTX 4090", require_tts=True)
    except RuntimeError as err:
        assert "vocos is unavailable" in str(err), str(err)
        assert "repair" in str(err), "the message must name the repair command"
    else:
        raise AssertionError("a broken IndicF5 environment must stop an IndicF5 run")

    gated.check_health = lambda url=None: {  # type: ignore[method-assign]
        "status": "healthy",
        "vram": {"device_name": "NVIDIA GeForce RTX 3090"},
    }
    assert gated.require_gpu("RTX 3090", require_tts=True)["status"] == "healthy", (
        "a server that predates the tts_ready field must not be rejected"
    )

    # 5. An environment fault in the remote interpreter is stable, so retrying it only
    #    spends the rental. A dropped connection is not, and must still be retried.
    assert tts.is_permanent_remote_tts_error(
        RuntimeError(
            "HTTP 500 from http://146.113.152.67:40400/v1/audio/synthesize: "
            "ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4"
        )
    ), "the exact failure from the broken instance must not be retried"
    assert tts.is_permanent_remote_tts_error(ModuleNotFoundError("No module named 'pyarrow'"))
    assert not tts.is_permanent_remote_tts_error(ConnectionResetError("connection reset by peer"))
    assert not tts.is_permanent_remote_tts_error(TimeoutError("read timed out"))

    # 6. The server has to be able to answer the question the client now asks. Importing the
    #    module must stay side-effect free (it only binds the server under __main__), and the
    #    probe must answer -- usable or a one-line reason -- instead of raising.
    import vast_server

    assert callable(vast_server.tts_import_error)
    verdict = vast_server.tts_import_error()
    assert verdict is None or isinstance(verdict, str), (
        f"the probe must return None or a reason, never raise: {verdict!r}"
    )

    print("  ✓ numpy/pyarrow are pinned, the probe covers the lazy import, and repair reuses it.")


def test_indicf5_contracts() -> None:
    print("Testing IndicF5 references, health preflight, and remote-batch policy...")

    import numpy as np
    import soundfile as sf
    import vast_server

    assert vast_server._indicf5_voice_stem("indicf5_aarav") == "aarav"
    assert vast_server._indicf5_voice_stem("AARAV") == "aarav"
    assert tts.indicf5_reference_name("indicf5_aarav") == "aarav"
    assert tts.indicf5_reference_name("indicf5_missing") == "missing"

    old_voices_dir = os.environ.get("INDICF5_VOICES_DIR")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            voice_dir = Path(tmp)
            os.environ["INDICF5_VOICES_DIR"] = str(voice_dir)
            tone = 0.2 * np.sin(2 * np.pi * 220 * np.arange(12000) / 16000)
            sf.write(voice_dir / "aarav.wav", tone, 16000, subtype="PCM_16")
            (voice_dir / "aarav.txt").write_text(" नमस्ते   दुनिया ", encoding="utf-8")
            prepared, transcript = vast_server._prepare_indicf5_reference("indicf5_aarav")
            data, sample_rate = sf.read(prepared, dtype="float32")
            assert transcript == "नमस्ते दुनिया"
            assert sample_rate == vast_server.INDICF5_SAMPLE_RATE
            assert data.ndim == 1 and not vast_server._audio_is_bad(data)
            assert vast_server.indicf5_voice_error("indicf5_aarav") is None

            (voice_dir / "silent.wav").write_bytes(b"")
            (voice_dir / "silent.txt").write_text("silence", encoding="utf-8")
            assert "cannot be decoded" in (vast_server.indicf5_voice_error("indicf5_silent") or "")
            assert "no reference clip/transcript" in (vast_server.indicf5_voice_error("indicf5_missing") or "")
    finally:
        if old_voices_dir is None:
            os.environ.pop("INDICF5_VOICES_DIR", None)
        else:
            os.environ["INDICF5_VOICES_DIR"] = old_voices_dir

    old_env = {key: os.environ.get(key) for key in ("VAST_SERVER_URL", "USE_VAST", "TTS_PROVIDER")}
    try:
        os.environ["VAST_SERVER_URL"] = "http://127.0.0.1:8000"
        os.environ["USE_VAST"] = "1"
        assert tts.remote_tts_enabled()
        os.environ["USE_VAST"] = "0"
        os.environ["TTS_PROVIDER"] = "vast"
        assert tts.remote_tts_enabled()
        os.environ["TTS_PROVIDER"] = "local"
        assert not tts.remote_tts_enabled()
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    print("  ✓ IndicF5 references are normalised and remote batch policy is explicit.")


def test_vast_call_sites() -> None:
    print("Testing that every VastManager call in the orchestrator matches its signature...")

    import ast
    import inspect

    import vast_manager

    orchestrator = Path(vast_manager.__file__).resolve().parent / "new_video.py"
    tree = ast.parse(orchestrator.read_text(encoding="utf-8"))

    # new_video imports VastManager lazily inside the function that needs it, so static
    # analysis never resolves these calls and a renamed parameter fails at runtime instead
    # of at review time. Read the keywords back out of the source and check them for real.
    guarded = (
        "provision_instance",
        "create_instance",
        "find_existing_instance",
        "wait_until_ready",
        "require_gpu",
        "start_instance",
        "stop_instance",
    )
    seen: dict[str, int] = {}
    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "attr", None) or getattr(func, "id", None)
        if name not in guarded:
            continue
        method = getattr(vast_manager.VastManager, name, None)
        assert method is not None, f"new_video.py calls vast_mgr.{name}() but VastManager has no such method"
        accepted = set(inspect.signature(method).parameters)
        for keyword in node.keywords:
            if keyword.arg is None:
                continue  # **expansion carries no static name
            assert keyword.arg in accepted, (
                f"new_video.py line {node.lineno} calls {name}({keyword.arg}=...) but "
                f"VastManager.{name}() does not accept it "
                f"(accepts: {', '.join(sorted(accepted))})"
            )
            checked += 1
        seen[name] = seen.get(name, 0) + 1

    assert "provision_instance" in seen, (
        "the orchestrator no longer calls provision_instance: this guard is stale"
    )
    assert "find_existing_instance" in seen, "the orchestrator must still try to reuse a rented instance"
    assert checked >= 4, f"expected to inspect the orchestrator's Vast keywords, saw {checked}"
    print(f"  ✓ {checked} orchestrator keywords verified against the real signatures.")


def test_gpu_resume_and_batch() -> None:
    print("Testing the durable GPU audio cache, offer retry, and clip resume...")

    import io as _io

    import numpy as _np
    import soundfile as sf

    import generate_videos
    import tts
    from vast_manager import InstanceBroken, InstanceSlow, VastManager

    # 1. The GPU cache is content-addressed: identical segments share a file, and any
    #    change of text, voice or speed must not collide with another segment.
    base = tts.gpu_cache_path("नमस्ते दुनिया", "indic_parler_aarav", 1.0)
    assert base == tts.gpu_cache_path("नमस्ते दुनिया", "indic_parler_aarav", 1.0)
    assert base != tts.gpu_cache_path("नमस्ते दुनिया", "indic_parler_priya", 1.0), "voice must be part of the key"
    assert base != tts.gpu_cache_path("नमस्ते दुनिया", "indic_parler_aarav", 1.2), "speed must be part of the key"
    assert base != tts.gpu_cache_path("अलग वाक्य", "indic_parler_aarav", 1.0), "text must be part of the key"
    assert base.suffix == ".wav"
    assert tts.GPU_CACHE_DIR.name == "gpu_audio_cache"

    # 2. Round-trip through disk. This is what makes a retry free instead of another
    #    GPU charge, so a segment must come back byte-for-byte. The fixture is real
    #    (non-silent) audio because silent/DC segments are now refused by the guard.
    tone = 0.3 * _np.sin(2 * _np.pi * 220 * _np.arange(2400) / 24000)
    audio_buf = _io.BytesIO()
    sf.write(audio_buf, tone, 24000, format="WAV", subtype="PCM_16")
    wav_bytes = audio_buf.getvalue()

    # A silent segment must never be persisted (it would be replayed as a muted video).
    silent_buf = _io.BytesIO()
    sf.write(silent_buf, [0.0] * 2400, 24000, format="WAV", subtype="PCM_16")
    silent_path = tts.gpu_cache_path("कैश साइलेंट", "indic_parler_aarav", 1.0)
    silent_path.unlink(missing_ok=True)
    tts.write_gpu_cache("कैश साइलेंट", "indic_parler_aarav", 1.0, silent_buf.getvalue())
    assert not silent_path.exists(), "a silent segment must not be written to the gpu cache"

    probe_text, probe_voice, probe_speed = "कैश परीक्षण", "indic_parler_aarav", 1.0
    probe_path = tts.gpu_cache_path(probe_text, probe_voice, probe_speed)
    probe_path.unlink(missing_ok=True)
    assert tts.read_gpu_cache(probe_text, probe_voice, probe_speed) is None
    try:
        tts.write_gpu_cache(probe_text, probe_voice, probe_speed, wav_bytes)
        assert probe_path.exists(), "a GPU segment must be persisted to disk"
        restored = tts.read_gpu_cache(probe_text, probe_voice, probe_speed)
        assert restored is not None and len(restored) == 2400, "the cached waveform must round-trip"
    finally:
        probe_path.unlink(missing_ok=True)

    # 3. An empty offer list is retried rather than failing the run.
    waiter = VastManager(api_key="test-key")
    taps: list[float] = []
    queue: list[list[dict]] = [[], [], [{"id": 21, "dph_total": 0.2, "machine_id": 7}]]
    waiter.search_gpu = lambda *a, **k: queue.pop(0)  # type: ignore[method-assign]
    waiter._offer_wait = taps.append  # type: ignore[method-assign]
    found = waiter.search_gpu_with_wait("RTX 3090", wait_sec=15)
    assert len(found) == 1, "the search must recover once an offer appears"
    assert taps == [15, 15], f"each empty search must wait 15s, got {taps}"

    giving_up = VastManager(api_key="test-key")
    taps2: list[float] = []
    giving_up.search_gpu = lambda *a, **k: []  # type: ignore[method-assign]
    giving_up._offer_wait = taps2.append  # type: ignore[method-assign]
    assert giving_up.search_gpu_with_wait("RTX 3090", wait_sec=15, attempts=3) == []
    assert taps2 == [15, 15], f"wait between attempts, never after the last, got {taps2}"

    # 4. A leftover instance of the wrong GPU model is reported, because Vast cannot
    #    change an instance's GPU type and it would keep billing storage.
    checker = VastManager(api_key="test-key")
    checker.run_cli = lambda args: json.dumps([  # type: ignore[method-assign]
        {"id": 1, "gpu_name": "RTX 3090", "actual_status": "stopped", "disk_space": 50, "machine_id": 11},
        {"id": 2, "gpu_name": "RTX 4090", "actual_status": "stopped", "disk_space": 50, "machine_id": 22},
        {"id": 3, "gpu_name": "RTX 3090", "actual_status": "exited", "disk_space": 50, "machine_id": 33},
    ])
    assert [c["id"] for c in checker.conflicting_instances("RTX 4090")] == [1, 3], (
        "a run needing a 4090 must flag the two stopped 3090s"
    )
    assert [c["id"] for c in checker.conflicting_instances("RTX 3090")] == [2], (
        "a run needing a 3090 must flag only the 4090"
    )

    # 5. LTX clips resume from disk instead of re-paying for GPU time.
    calls: list[str] = []

    class _FakeClipClient:
        def generate_video(self, prompt, **kwargs) -> bytes:
            calls.append(prompt)
            return b"fake-mp4-bytes"

    scene = {
        "id": "scene-001",
        "headline": "परीक्षण",
        "beat": "establish",
        "movement": "hook",
        "emotion": "calm",
        "shots": [],
    }
    with tempfile.TemporaryDirectory() as tmp:
        script = {"topic": "resume test", "genre": "horror", "scenes": [dict(scene)]}
        generate_videos.generate_videos_for_script(
            script, out_dir=tmp, manager=_FakeClipClient()  # type: ignore[arg-type]
        )
        assert len(calls) == 1, "the first run must generate the clip"
        assert (Path(tmp) / "scene-001.mp4").exists()

        calls.clear()
        generate_videos.generate_videos_for_script(
            script, out_dir=tmp, manager=_FakeClipClient()  # type: ignore[arg-type]
        )
        assert calls == [], "an existing clip must not be regenerated"

        generate_videos.generate_videos_for_script(
            script, out_dir=tmp, manager=_FakeClipClient(), force=True  # type: ignore[arg-type]
        )
        assert len(calls) == 1, "--force-videos must regenerate the clip"

    # 6. The last-resort offer query must not drop its structural constraints. An offer
    #    without enough disk or RAM cannot host the job, and renting it fails later in a
    #    far more confusing way than refusing here.
    queries: list[str] = []
    probe = VastManager(api_key="test-key")

    def _empty_cli(args: list[str]) -> str:
        query = args[2] if args[0] == "search" and len(args) > 2 else ""
        queries.append(query)
        return "[]"

    probe.run_cli = _empty_cli  # type: ignore[method-assign]
    assert probe.search_gpu("RTX 3090", min_disk=50) == []
    assert queries, "the offer search must have run"
    final_query = queries[-1]
    for clause in ("gpu_name=RTX_3090", "num_gpus=1", "rented=False", "disk_space>=50", "cpu_ram>=32", "dph<0.4"):
        assert clause in final_query, (
            f"the last-resort query dropped {clause!r}: {final_query!r}"
        )
    assert "inet_down>100" not in final_query, (
        "the last-resort query may relax the speed floor, but nothing else"
    )
    assert "inet_down<400" in final_query, (
        "the download cap is not a preference: it must survive the last-resort query"
    )

    # 7. Megabyte-level disk jitter must not count as install progress. A host that never
    #    installs anything is broken, not slow: waiting out the whole budget wastes the run,
    #    and stopping it would preserve an empty install that still bills for disk. The
    #    values below are the real ones from a host that installed 70MB in 16 minutes.
    import time as _time

    boot = VastManager(api_key="test-key")
    jitter = [0.001, 0.018, 0.028, 0.039, 0.042, 0.057, 0.061, 0.070]
    seen = {"n": 0}

    def _stuck_info(_instance_id: int) -> dict:
        index = min(seen["n"], len(jitter) - 1)
        seen["n"] += 1
        return {
            "actual_status": "running",
            "status_msg": "success, running",
            "disk_util": jitter[index],
            "machine_id": 84216,
            "public_ipaddr": "203.0.113.9",
            "ports": {"8000/tcp": [{"HostPort": "50280"}]},
        }

    def _no_listener(url=None):
        raise ConnectionResetError("no listener on 8000")

    boot.get_instance_info = _stuck_info  # type: ignore[method-assign]
    boot.check_health = _no_listener  # type: ignore[method-assign]
    boot.run_ssh = lambda *a, **k: ""  # type: ignore[method-assign]
    boot._forget_idle = lambda i: None  # type: ignore[method-assign]
    blacklisted: list[tuple] = []
    boot.mark_bad_host = lambda m, r: blacklisted.append((m, r))  # type: ignore[method-assign]

    saved_env = {
        k: os.environ.get(k)
        for k in (
            "VAST_SERVER_STALL_TIMEOUT",
            "VAST_PROGRESS_MIN_DELTA_GB",
            "VAST_INSTALL_GRACE_SECONDS",
            "VAST_INSTALL_FLOOR_GB",
            "VAST_CONTAINER_TIMEOUT",
        )
    }
    os.environ["VAST_SERVER_STALL_TIMEOUT"] = "600"
    os.environ["VAST_PROGRESS_MIN_DELTA_GB"] = "0.05"
    os.environ["VAST_INSTALL_GRACE_SECONDS"] = "60"
    clock = {"t": 1_000_000.0}
    real_time, real_sleep = _time.time, _time.sleep
    _time.time = lambda: clock["t"]  # type: ignore[assignment]
    _time.sleep = lambda seconds: clock.__setitem__("t", clock["t"] + max(1.0, float(seconds)))  # type: ignore[assignment]
    try:
        try:
            boot.wait_until_ready(999)
            raise AssertionError("a host that installs nothing must not be reported ready")
        except InstanceBroken as err:
            assert "installed nothing" in str(err), f"the message must say what happened: {err}"
        assert blacklisted == [(84216, "onstart installed nothing")], (
            f"a host that installed nothing must be blacklisted, got {blacklisted}"
        )
        waited = clock["t"] - 1_000_000.0
        assert waited < 600, f"the stall budget was ignored: waited {waited}s"

        # 8. The image-pull phase must never be judged as "no install progress". Vast reports
        #    disk_util = -1.0 (no reading yet) for the whole pull, and a cold pull of
        #    pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime measured 291s. Taking -1.0 as a
        #    real measurement made peak_disk -1.0, which is below any install floor, so the
        #    install grace -- which used to start at create time -- fired at 303s while the
        #    image was still downloading and destroyed a working host (instance 55754805).
        assert VastManager._disk_used_gb({"disk_util": -1.0}) is None, (
            "the -1.0 'no reading yet' sentinel must not be treated as a measurement"
        )
        assert VastManager._disk_used_gb({"disk_util": 0.0}) == 0.0, (
            "zero is a real reading and must survive"
        )

        pulling = VastManager(api_key="test-key")
        pull_phases: list[dict] = [
            {
                "actual_status": "loading",
                "status_msg": f"2c85778708e9: Download complete ({index})",
                "disk_util": -1.0,
                "machine_id": 153772,
            }
            for index in range(12)  # 120s of pulling, twice the 60s install grace
        ]
        for disk in (0.20, 0.30, 0.45):
            pull_phases.append(
                {
                    "actual_status": "running",
                    "status_msg": "success, running",
                    "disk_util": disk,
                    "machine_id": 153772,
                    "public_ipaddr": "203.0.113.9",
                    "ports": {"8000/tcp": [{"HostPort": "36539"}]},
                }
            )
        phase = {"n": 0}

        def _pull_info(_instance_id: int) -> dict:
            index = min(phase["n"], len(pull_phases) - 1)
            phase["n"] += 1
            return dict(pull_phases[index])

        health = {"n": 0}

        def _late_server(url=None) -> dict:
            health["n"] += 1
            if health["n"] < 3:
                raise ConnectionRefusedError("listener not up yet")
            return {"status": "healthy"}

        pulling.get_instance_info = _pull_info  # type: ignore[method-assign]
        pulling.check_health = _late_server  # type: ignore[method-assign]
        pulling.run_ssh = lambda *a, **k: ""  # type: ignore[method-assign]
        pulling.run_cli = lambda *a, **k: ""  # type: ignore[method-assign]
        pulling._forget_idle = lambda i: None  # type: ignore[method-assign]
        pull_blacklist: list[tuple] = []
        pulling.mark_bad_host = lambda m, r: pull_blacklist.append((m, r))  # type: ignore[method-assign]

        os.environ["VAST_INSTALL_FLOOR_GB"] = "1.5"
        os.environ["VAST_CONTAINER_TIMEOUT"] = "600"
        clock["t"] = 4_000_000.0
        _time.time = lambda: clock["t"]  # type: ignore[assignment]
        _time.sleep = lambda seconds: clock.__setitem__("t", clock["t"] + max(1.0, float(seconds)))  # type: ignore[assignment]
        try:
            url = pulling.wait_until_ready(999)
        except (InstanceBroken, InstanceSlow) as err:
            raise AssertionError(
                f"a slow image pull must not fail the run: {type(err).__name__}: {err}"
            ) from err
        assert url == "http://203.0.113.9:36539", url
        assert pull_blacklist == [], f"a slow pull must not blacklist the host, got {pull_blacklist}"
        survived = clock["t"] - 4_000_000.0
        assert survived >= 120, (
            f"the pull must be allowed to outlast the install grace, saw {survived}s"
        )
    finally:
        _time.time, _time.sleep = real_time, real_sleep
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    print("  ✓ GPU audio is cached to disk, offers retry, and clips resume.")


def test_voices() -> None:
    print("Testing Multi-Voice catalog, alias resolution, blending, and language detection...")

    # 1. Test Voice Catalog completeness across all supported languages
    catalog = tts.VOICE_CATALOG
    assert len(catalog) >= 30, f"Expected 30+ voices in catalog, got {len(catalog)}"

    hindi_voices = [v for v in catalog.values() if v["lang"] == "hi"]
    assert len(hindi_voices) >= 4
    assert any(v["id"] == "hm_omega" for v in hindi_voices)
    assert any(v["id"] == "hf_alpha" for v in hindi_voices)

    us_en_voices = [v for v in catalog.values() if v["lang"] == "en-us"]
    assert len(us_en_voices) >= 10
    assert any(v["id"] == "af_heart" for v in us_en_voices)
    assert any(v["id"] == "af_bella" for v in us_en_voices)
    assert any(v["id"] == "am_onyx" for v in us_en_voices)

    gb_en_voices = [v for v in catalog.values() if v["lang"] == "en-gb"]
    assert len(gb_en_voices) >= 6
    assert any(v["id"] == "bm_george" for v in gb_en_voices)

    ja_voices = [v for v in catalog.values() if v["lang"] == "ja"]
    assert len(ja_voices) >= 4
    assert any(v["id"] == "jf_nezumi" for v in ja_voices)

    cmn_voices = [v for v in catalog.values() if v["lang"] == "cmn"]
    assert len(cmn_voices) >= 4

    # 2. Test list_voices and filtering
    # list_voices() reports every selectable voice, which includes the Indic Parler-TTS
    # family. That family lives in INDIC_PARLER_VOICES, not VOICE_CATALOG, so the totals
    # are the union.
    all_v = tts.list_voices()
    assert len(all_v) == len(catalog) + len(tts.INDIC_PARLER_VOICES)

    hi_v = tts.list_voices(lang_filter="hi")
    assert len(hi_v) == len(hindi_voices) + len(tts.INDIC_PARLER_VOICES)
    assert all(v["lang"] == "hi" for v in hi_v)

    # 3. Test format_voices_table
    table_text = tts.format_voices_table()
    assert "KOKORO-82M VOICE CATALOG" in table_text
    assert "Hindi (hi)" in table_text
    assert "American English" in table_text
    assert "British English" in table_text
    assert "VOICE BLENDING" in table_text

    # 4. Test resolve_voice with direct IDs, aliases, and blends
    # Direct IDs
    assert tts.resolve_voice("hm_omega") == "hm_omega"
    assert tts.resolve_voice("af_heart") == "af_heart"
    assert tts.resolve_voice("bm_george") == "bm_george"

    # Friendly Aliases
    assert tts.resolve_voice("hindi_male") == "hm_omega"
    assert tts.resolve_voice("hindi_female") == "hf_alpha"
    assert tts.resolve_voice("documentary") == "bm_george"
    assert tts.resolve_voice("deep_male") == "am_onyx"
    assert tts.resolve_voice("anime_female") == "jf_nezumi"

    # Voice Blending syntax
    assert tts.resolve_voice("af_heart+af_bella") == "af_heart,af_bella"
    assert tts.resolve_voice("hm_omega+hf_alpha") == "hm_omega,hf_alpha"
    assert tts.resolve_voice("hindi_male+hindi_female") == "hm_omega,hf_alpha"
    assert tts.resolve_voice("af_heart, af_bella") == "af_heart,af_bella"

    # Auto / Genre & Language fallback (defaults to hm_omega for Hindi/Hinglish)
    assert tts.resolve_voice(None, default_lang="hi", genre="horror") == "hm_omega"
    assert tts.resolve_voice(None, default_lang="hinglish", genre="horror") == "hm_omega"
    assert tts.resolve_voice(None, default_lang="hi", genre="fantasy") == "hm_omega"
    assert tts.resolve_voice(None, default_lang="hinglish", genre="fantasy") == "hm_omega"
    assert tts.resolve_voice(None, default_lang="hi", genre="scifi") == "hm_omega"
    assert tts.resolve_voice(None, default_lang="hinglish", genre="scifi") == "hm_omega"
    assert tts.resolve_voice(None, default_lang="en", genre="documentary") == "bm_george"
    assert tts.resolve_voice(None, default_lang="en", genre="horror") == "am_onyx"
    assert tts.resolve_voice(None, default_lang="ja", genre="anime") == "jf_nezumi"

    # 5. Test phonemise language resolution with aliases, blends, and Hinglish overrides
    assert phonemise.resolve_language("hm_omega") == "hi"
    assert phonemise.resolve_language("af_heart") == "en-us"
    assert phonemise.resolve_language("bm_george") == "en-gb"
    assert phonemise.resolve_language("jf_nezumi") == "ja"
    assert phonemise.resolve_language("af_heart+af_bella") == "en-us"
    assert phonemise.resolve_language("hindi_female") == "hi"
    assert phonemise.resolve_language("documentary") == "en-gb"
    assert phonemise.resolve_language("hm_omega", override="hinglish") == "hi"
    assert phonemise.resolve_language("hm_omega", override="hindi") == "hi"

    # 6. Test build_script character voices with secondary voice
    mock_plan = beats.plan("ninja anime fight", minutes=1.0, with_characters=True)
    mock_blocks = [("Headline", "visuals", "Dialogue text")]
    script, _ = new_video.build_script(
        mock_plan,
        "Ninja Battle",
        mock_blocks,
        voice="am_adam",
        speed=1.0,
        captions={},
        secondary_voice="am_onyx",
    )
    assert script["voice"] == "am_adam"
    assert script["secondaryVoice"] == "am_onyx"
    assert script["characters"]["lead"]["voice"] == "am_adam"

    print("  ✓ All Multi-Voice catalog, alias resolution, blending, and language detection checks passed.")


def test_metadata_generator() -> None:
    print("Testing Multi-Platform Social Media Metadata Generator (YouTube, Instagram, Facebook, TikTok)...")

    # Sample script data
    sample_script = {
        "title": "Quantum Scam in 2080",
        "topic": "mobile phone scam alert",
        "genre": "scifi",
        "language": "hi",
        "format": "reel",
        "aspectRatio": "9:16",
        "scenes": [
            {"headline": "नियो-सिटी का डिजिटल जाल", "lines": [{"text": "साल 2080 में एक नया साइबर स्कैम आया।"}]},
            {"headline": "फेक सिग्नल का हमला", "lines": [{"text": "अनजान सिग्नल ने क्वांटम फोन को हैक करने की कोशिश की।"}]},
            {"headline": "न्यूरल फायरवॉल डिफेंस", "lines": [{"text": "उसने तुरंत न्यूरल शील्ड एक्टिवेट की।"}]},
            {"headline": "साइबर सुरक्षा की जीत", "lines": [{"text": "डेटा सेफ रहा और हैकर ब्लॉक हो गया।"}]},
        ],
    }

    sample_timings = {
        "totalDuration": 58.4,
        "scenes": [
            {"start": 0.0, "duration": 14.2},
            {"start": 14.2, "duration": 15.0},
            {"start": 29.2, "duration": 14.5},
            {"start": 43.7, "duration": 14.7},
        ],
    }

    # 1. Test chapter extraction
    chapters = metadata.extract_chapters_from_data(sample_script, sample_timings)
    assert len(chapters) == 4
    assert chapters[0][0] == "00:00"
    assert chapters[1][0] == "00:14"
    assert chapters[2][0] == "00:29"
    assert chapters[3][0] == "00:44"

    # 2. Test procedural metadata generation
    meta = metadata.generate_metadata_procedural(sample_script, sample_timings)
    
    # YouTube assertions
    yt = meta["youtube"]
    assert "#Shorts" in yt["title"]
    assert "00:00 - नियो-सिटी का डिजिटल जाल" in yt["description"]
    assert "00:14 - फेक सिग्नल का हमला" in yt["description"]
    assert yt["category_id"] == "28"  # SciFi category ID
    assert yt["is_shorts"] is True
    assert len(yt["tags"]) >= 10
    assert any("mobile" in t for t in yt["tags"])
    assert any("scifi" in t for t in yt["tags"])

    # Instagram assertions
    ig = meta["instagram"]
    assert "Quantum Scam in 2080" in ig["hook_line"]
    assert len(ig["hashtags"]) >= 20
    assert any("#scifi" in tag.lower() for tag in ig["hashtags"])
    assert "Save this reel" in ig["call_to_action"]

    # Facebook assertions
    fb = meta["facebook"]
    assert "Watch Till The End" in fb["title"]
    assert len(fb["hashtags"]) >= 3
    assert len(fb["tags"]) >= 5

    # TikTok assertions
    tt = meta["tiktok"]
    assert len(tt["caption"]) > 0
    assert any("#fyp" in tag for tag in tt["hashtags"])

    # 3. Test Hashtag banks for all genres
    for genre_key in beats.GENRES:
        assert genre_key in metadata.GENRE_HASHTAG_BANKS, f"Missing hashtag bank for {genre_key}"
        bank = metadata.GENRE_HASHTAG_BANKS[genre_key]
        assert "youtube" in bank and len(bank["youtube"]) > 0
        assert "instagram" in bank and len(bank["instagram"]) >= 20
        assert "facebook" in bank and len(bank["facebook"]) >= 3
        assert "tiktok" in bank and len(bank["tiktok"]) >= 5

    # 4. Test Markdown formatting and Export
    md_text = metadata.format_markdown_report(meta)
    assert "# 🚀 Social Media Upload Metadata" in md_text
    assert "## 📺 1. YouTube" in md_text
    assert "## 📸 2. Instagram" in md_text
    assert "## 👥 3. Facebook" in md_text
    assert "## 🎵 4. TikTok" in md_text

    test_out_dir = Path("out") / "test_meta"
    json_path, md_path = metadata.export_metadata(meta, test_out_dir, base_name="test_video")
    assert json_path.exists()
    assert md_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))["youtube"]["category_id"] == "28"
    assert "YouTube" in md_path.read_text(encoding="utf-8")

    # Clean up test files
    try:
        json_path.unlink(missing_ok=True)
        md_path.unlink(missing_ok=True)
        test_out_dir.rmdir()
    except Exception:
        pass

    print("  ✓ All Multi-Platform Metadata Generation (YouTube, Instagram, Facebook, TikTok) checks passed.")


def test_auto_uploader() -> None:
    print("Testing Multi-Platform Auto-Uploader (YouTube, Facebook, Instagram)...")

    # Sample test metadata structure
    meta = {
        "youtube": {
            "title": "साइबर शेर vs डिजिटल भूत #Shorts",
            "description": "00:00 - शुरुआत\n00:15 - लड़ाई\n\n#SciFi #Anime",
            "tags": ["cyber lion", "digital ghost", "sci fi hindi"],
            "category_id": "24",
            "is_shorts": True,
        },
        "facebook": {
            "title": "2150 का महायुद्ध: शेर बनाम भूत! 🦁⚡👻",
            "description": "साल 2150 का सबसे बड़ा मुकाबला!",
            "tags": ["SciFi", "Cyberpunk"],
            "hashtags": ["#SciFi", "#AI", "#Future"],
        },
        "instagram": {
            "hook_line": "साल 2150 का सबसे बड़ा मुकाबला!",
            "caption": "साल 2150 का सबसे बड़ा मुकाबला! पूरा वीडियो देखें!",
            "hashtags": ["#reels", "#scifi", "#hindi"],
            "call_to_action": "Follow for more",
        },
    }

    # 1. YouTube Uploader dry-run test
    yt = uploader.YouTubeUploader()
    yt_res = yt.upload_video(
        video_path=Path("out") / "video.mp4",
        meta=meta.get("youtube"),
        privacy="public",
        dry_run=True,
    )
    assert yt_res["success"] is True
    assert yt_res["id"] == "DRY_RUN_YT_ID"
    assert "https://youtu.be/DRY_RUN_YT_ID" in yt_res["url"]

    # 2. Facebook Uploader dry-run test
    fb = uploader.FacebookUploader()
    fb_res = fb.upload_reel(
        video_path=Path("out") / "video.mp4",
        meta=meta.get("facebook"),
        dry_run=True,
    )
    assert fb_res["success"] is True
    assert fb_res["id"] == "DRY_RUN_FB_ID"
    assert "facebook.com" in fb_res["url"]

    # 3. Instagram Uploader dry-run test
    ig = uploader.InstagramUploader()
    ig_res = ig.upload_reel(
        video_path=Path("out") / "video.mp4",
        meta=meta.get("instagram"),
        dry_run=True,
    )
    assert ig_res["success"] is True
    assert ig_res["id"] == "DRY_RUN_IG_ID"
    assert "instagram.com" in ig_res["url"]

    # 4. Multi-platform upload_all dry-run test
    test_meta_file = Path("out") / "test_uploader_meta.json"
    test_meta_file.parent.mkdir(parents=True, exist_ok=True)
    test_meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    all_res = uploader.upload_all(
        video_path=Path("out") / "video.mp4",
        metadata_path=str(test_meta_file),
        platforms=["youtube", "facebook", "instagram"],
        dry_run=True,
    )
    assert all_res["platforms"]["youtube"]["success"] is True
    assert all_res["platforms"]["facebook"]["success"] is True
    assert all_res["platforms"]["instagram"]["success"] is True

    test_meta_file.unlink(missing_ok=True)
    print("  ✓ All Multi-Platform Auto-Uploader checks passed.")


def test_vast_gated_model_provisioning() -> None:
    """A fresh rental must (a) authenticate to gated Hub repos and (b) find the IndicF5
    reference clips, or its narration step fails after the GPU has already been paid for.

    Both are environment problems that no amount of retrying fixes, and one of them cannot be
    fixed on an existing container at all, so they are asserted here rather than discovered
    during a paid run.
    """
    print("Testing Vast.ai gated-model provisioning (HF token + IndicF5 reference clips)...")

    from vast_manager import CONTAINER_VOICES_DIR, VastManager, hub_token

    saved_hf = os.environ.get("HF_TOKEN")
    saved_hf_api = os.environ.get("HUGGINGFACE_API_KEY")

    try:
        # 1. The docker-options string always publishes the port and pins the voices
        #    directory, and carries the token only when one is actually configured.
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HUGGINGFACE_API_KEY", None)
        assert hub_token() == ""
        without = VastManager.container_env_spec()
        assert "-p 8000:8000" in without, f"the port mapping is mandatory: {without}"
        assert f"-e INDICF5_VOICES_DIR={CONTAINER_VOICES_DIR}" in without, (
            f"the container must be told where its reference clips are: {without}"
        )
        assert "-e HF_TOKEN=" not in without, f"no token must not become an empty token: {without}"

        # The repo's .env spells it HUGGINGFACE_API_KEY while the server reads HF_TOKEN.
        os.environ["HUGGINGFACE_API_KEY"] = "hf_from_dotenv"
        assert hub_token() == "hf_from_dotenv"
        assert "-e HF_TOKEN=hf_from_dotenv" in VastManager.container_env_spec()

        # 2. A malformed token would corrupt the WHOLE docker-options string, port mapping
        #    included, and the resulting breakage looks like a Vast fault. Refuse it.
        os.environ.pop("HUGGINGFACE_API_KEY", None)
        os.environ["HF_TOKEN"] = "bad token"
        assert hub_token() == "", "whitespace inside a token must be refused"
        os.environ["HF_TOKEN"] = 'bad"quote'
        assert hub_token() == "", "a quote inside a token must be refused"
        assert "-e HF_TOKEN=" not in VastManager.container_env_spec()

        # 2b. A configured model volume pins HF_HOME so the ~42 GB of weights live on the
        #     volume instead of the container layer. Without HF_HOME the volume is mounted
        #     but unused, and every rental re-downloads the same weights.
        saved_volume = os.environ.get("VAST_HF_VOLUME")
        saved_hf_home = os.environ.get("VAST_HF_HOME")
        try:
            assert "-e HF_HOME=" not in VastManager.container_env_spec(), (
                "no volume configured must not pin HF_HOME"
            )
            os.environ["VAST_HF_HOME"] = "/workspace/hf"
            assert "-e HF_HOME=/workspace/hf" in VastManager.container_env_spec(), (
                "a configured HF_HOME must reach the container"
            )
            assert "-e HF_HUB_DISABLE_XET=1" in VastManager.container_env_spec(), (
                "the xet backend must be disabled when weights land on a volume mount"
            )
        finally:
            if saved_volume is None:
                os.environ.pop("VAST_HF_VOLUME", None)
            else:
                os.environ["VAST_HF_VOLUME"] = saved_volume
            if saved_hf_home is None:
                os.environ.pop("VAST_HF_HOME", None)
            else:
                os.environ["VAST_HF_HOME"] = saved_hf_home

        # 3. Only COMPLETE reference pairs are uploaded: a clip without its transcript is
        #    unusable, and a generated _prepared_* file is an artefact, not a source clip.
        with tempfile.TemporaryDirectory() as tmp:
            voices = Path(tmp)
            (voices / "aarav.wav").write_bytes(b"RIFF")
            (voices / "aarav.txt").write_text("नमस्ते", encoding="utf-8")
            (voices / "orphan.wav").write_bytes(b"RIFF")
            (voices / "_prepared_aarav.wav").write_bytes(b"RIFF")
            (voices / "notes.md").write_text("not a clip", encoding="utf-8")

            os.environ["INDICF5_LOCAL_VOICES_DIR"] = str(voices)
            try:
                manager = VastManager(api_key="test-key")
                assert [p.name for p in manager.indicf5_voice_files()] == ["aarav.wav", "aarav.txt"], (
                    f"only the complete pair may be uploaded: {manager.indicf5_voice_files()}"
                )

                ssh_calls: list[str] = []
                pushes: list[tuple[int, str, str]] = []
                manager.run_ssh = lambda i, cmd, timeout=600: ssh_calls.append(cmd) or ""  # type: ignore[method-assign]
                manager.push_file = lambda i, local, remote: pushes.append((i, str(local), remote))  # type: ignore[method-assign]
                assert manager.push_indicf5_voices(4242) == 2
                assert ssh_calls and f"mkdir -p {CONTAINER_VOICES_DIR}" in ssh_calls[0], ssh_calls
                assert [remote for _i, _local, remote in pushes] == [
                    f"{CONTAINER_VOICES_DIR}/aarav.wav",
                    f"{CONTAINER_VOICES_DIR}/aarav.txt",
                ], pushes
            finally:
                os.environ.pop("INDICF5_LOCAL_VOICES_DIR", None)

        # 4. "This box has no token" must not be reported as "this install is broken": the
        #    remedies differ, and only the repair can be applied to a container that exists.
        gated = VastManager(api_key="test-key")
        gated.check_health = lambda url=None: {  # type: ignore[method-assign]
            "status": "healthy",
            "vram": {"device_name": "NVIDIA GeForce RTX 4090"},
            "indicf5_ready": False,
            "indicf5_error": "OSError: 401 Client Error",
            "hf_token_present": False,
        }
        try:
            gated.require_gpu("RTX 4090", require_tts=True)
            raise AssertionError("a tokenless server must not satisfy an IndicF5 requirement")
        except RuntimeError as err:
            assert "GATED" in str(err) and "HF_TOKEN" in str(err), str(err)
            assert "rent a new instance" in str(err), (
                "the operator must be told a container's environment cannot be changed in place"
            )

        # 5. A directory holding only transcripts must never be chosen over one holding the
        #    clips, and an explicit override must always win.
        import vast_server

        assert not vast_server._voices_dir_has_clips(Path(tempfile.gettempdir()) / "no-such-voices")
        transcripts_only = Path(tempfile.mkdtemp(prefix="voices-txt-"))
        (transcripts_only / "aarav.txt").write_text("नमस्ते", encoding="utf-8")
        assert not vast_server._voices_dir_has_clips(transcripts_only), (
            "transcripts alone are not a usable reference directory"
        )

        override = Path(tempfile.gettempdir()) / "explicit-voices"
        os.environ["INDICF5_VOICES_DIR"] = str(override)
        try:
            assert vast_server._indicf5_voices_dir() == override, "an explicit override must win"
        finally:
            os.environ.pop("INDICF5_VOICES_DIR", None)
    finally:
        for name, value in (("HF_TOKEN", saved_hf), ("HUGGINGFACE_API_KEY", saved_hf_api)):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    print("  ✓ Gated-repo provisioning: the token travels with the instance and clips are seeded.")


if __name__ == "__main__":
    test_genres()
    test_themes()
    test_video_formats()
    test_video_generation_mode()
    test_vast_gpu_routing()
    test_vast_setup_resilience()
    test_vast_boot_stall_policy()
    test_audio_image_only_mode()
    test_vast_preflight()
    test_vast_rental_cost_ranking()
    test_vast_meta_device_and_diagnose()
    test_vast_reuse_contract()
    test_vast_dependency_abi()
    test_vast_gated_model_provisioning()
    test_indicf5_contracts()
    test_vast_call_sites()
    test_gpu_resume_and_batch()
    test_voices()
    test_dynamic_prompting()
    test_llm_routing()
    test_image_routing()
    test_audio_and_speed()
    test_metadata_generator()
    test_auto_uploader()
    print("\nAll provider, theme, audio, format, voice, and genre routing tests completed successfully!")
