"""Test LLM and image provider routing and prompt construction.

Run with:  .venv\\Scripts\\python.exe scripts\\test_providers.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import beats
import generate_images
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
    assert "Conversational Spoken Hindi & Modern Loanword Rules" in hinglish_prompt
    assert "मोबाइल फोन" in hinglish_prompt
    assert "चल दूरभाष" in hinglish_prompt
    assert "कंप्यूटर" in hinglish_prompt
    assert "संगणक" in hinglish_prompt

    # 2. Test Devanagari Hindi prompt rules
    plan_dict_hindi = beats.plan("AI and Quantum Computer future", minutes=1.0, genre="scifi", language="hi")
    hindi_prompt = new_video.build_prompt(
        plan_dict_hindi,
        voice_style="cyberpunk thriller",
        genre_profile=profile,
    )
    assert "Conversational Spoken Hindi & Modern Loanword Rules" in hindi_prompt
    assert "मोबाइल फोन" in hindi_prompt
    assert "चल दूरभाष" in hindi_prompt
    assert "कंप्यूटर" in hindi_prompt
    assert "संगणक" in hindi_prompt
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

        # Local SD 1.5 + LCM image routing
        prov, key, model = generate_images.detect_provider("local")
        assert prov == "local" and key == "local" and "LCM" in model

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
    assert g_horror.default_speed == 0.88
    assert g_horror.default_wpm == 100

    g_motivational = beats.get_genre("motivational")
    assert g_motivational.default_speed == 1.05
    assert g_motivational.default_wpm == 130

    g_fantasy = beats.get_genre("fantasy")
    assert g_fantasy.default_speed == 0.94
    assert g_fantasy.default_wpm == 115

    # 5. Test beats.plan integration with audio cues and ambience
    plan_fantasy = beats.plan("ai dekho son pari", minutes=1.0, genre="fantasy")
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
    assert captions_reel["bottom"] == 0.18
    assert captions_reel["fontSize"] == 71

    captions_land = new_video.caption_options("subtitle", True, format_profile=fmt_land)
    assert captions_land["bottom"] == 0.08
    assert captions_land["fontSize"] == 62

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
    all_v = tts.list_voices()
    assert len(all_v) == len(catalog)

    hi_v = tts.list_voices(lang_filter="hi")
    assert len(hi_v) == len(hindi_voices)
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


if __name__ == "__main__":
    test_genres()
    test_themes()
    test_video_formats()
    test_voices()
    test_dynamic_prompting()
    test_llm_routing()
    test_image_routing()
    test_audio_and_speed()
    test_metadata_generator()
    test_auto_uploader()
    print("\nAll provider, theme, audio, format, voice, and genre routing tests completed successfully!")
