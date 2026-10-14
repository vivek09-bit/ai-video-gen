"""Verify the story planner always produces a complete arc.

Beats are visual grammar, not a story. Cycling them is how a long video used to
loop the same six shots and never actually end, so these invariants are the thing
that stops a video coming out half-finished at any length.

Run with:  .venv\\Scripts\\python.exe scripts\\test_story.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

import beats  # noqa: E402

ESSENTIAL = {"setup", "inciting", "climax", "resolution"}
OPENERS = {"hook", "setup"}
CLOSERS = {"resolution", "button"}

# ---------------------------------------------------------------------------
# The movement table itself
# ---------------------------------------------------------------------------
print(f"{'movement':<12} {'share':>6} {'beat':<10} duty")
print("-" * 92)
for movement in beats.MOVEMENTS.values():
    print(f"{movement.key:<12} {movement.weight:>6.2f} {movement.beat:<10} {movement.duty}")

total = sum(m.weight for m in beats.MOVEMENTS.values())
print(f"\nweights sum to {total:.2f}")
assert abs(total - 1.0) < 1e-9, f"movement weights must sum to 1.0, got {total}"
for key in ESSENTIAL:
    assert key in beats.MOVEMENTS, f"essential movement {key} is not defined"
assert ESSENTIAL <= set(beats.MOVEMENT_SETS[0][1]), "essential movements missing from standard story set"

# ---------------------------------------------------------------------------
# Every length must be a complete story
# ---------------------------------------------------------------------------
print()
print(f"{'length':>8} {'scenes':>7}  story arc")
print("-" * 92)
for minutes in (0.25, 0.5, 1, 2, 3, 5, 10, 20, 30, 60, 120):
    planned = beats.plan(topic="test topic", minutes=minutes)
    scenes = planned["scenes"]

    order: list[str] = []
    for scene in scenes:
        key = scene["_story"]
        if not order or order[-1] != key:
            order.append(key)

    counts: dict[str, int] = {}
    for scene in scenes:
        counts[scene["_story"]] = counts.get(scene["_story"], 0) + 1

    assert len(scenes) >= beats.MIN_SCENES, f"{minutes} min produced {len(scenes)} scenes"
    if len(scenes) >= 7:
        assert ESSENTIAL <= set(order), f"{minutes} min is missing {ESSENTIAL - set(order)}"
    assert order[0] in OPENERS, f"{minutes} min opens on {order[0]}"
    if len(scenes) >= 3:
        assert order[-1] in CLOSERS, f"{minutes} min ends on {order[-1]}"
    assert sum(counts.values()) == len(scenes), f"{minutes} min split does not total"
    assert all(
        scene.get("_storyNote") for scene in scenes
    ), f"{minutes} min has a scene with no story duty"

    detail = " ".join(f"{k}={v}" for k, v in counts.items())
    print(f"{minutes:>7g}m {len(scenes):>7}  {' -> '.join(order)}")
    print(f"{'':>17}  {detail}")

# ---------------------------------------------------------------------------
# The allocator must total exactly, without drifting a scene
# ---------------------------------------------------------------------------
for scene_count in range(0, 400):
    movements = beats.movements_for(max(scene_count, beats.MIN_SCENES))
    counts = beats.allocate(scene_count, movements)
    # Every movement keeps at least one scene, so the floor is the movement count.
    expected = max(scene_count, len(movements))
    assert sum(counts) == expected, f"allocate({scene_count}) totalled {sum(counts)}"
    assert all(c >= 1 for c in counts), f"allocate({scene_count}) gave a movement no scene"

for scene_count in range(beats.MIN_SCENES, 400):
    story = beats.story_plan(scene_count)
    assert len(story) == scene_count, f"story_plan({scene_count}) returned {len(story)}"
    assert [m.key for m, _b, _d in story] == [
        m.key for m, _b, _d in beats.story_plan(scene_count)
    ], "story_plan is not deterministic"
print("\nallocator totals exactly and deterministically for 0-399 scenes.")

# ---------------------------------------------------------------------------
# Scenes inside one movement must not be given the same instruction
# ---------------------------------------------------------------------------
story = beats.story_plan(25)
rising = [duty for movement, _beat, duty in story if movement.key == "rising"]
assert len(rising) > 1, "expected a multi-scene rising movement at 25 scenes"
assert len(set(rising)) == len(rising), f"rising duties repeat: {len(set(rising))} unique"
assert "complication 1 of" in rising[0] and f"complication {len(rising)} of" in rising[-1], rising
print(f"25 scenes: {len(rising)} distinct rising instructions, escalating 1..{len(rising)}.")

# Rising should not look like the same shot repeated.
story = beats.story_plan(25)
rising_beats = [beat for movement, beat, _d in story if movement.key == "rising"]
assert len(set(rising_beats)) > 1, f"rising uses one visual beat only: {set(rising_beats)}"
print(f"rising alternates visuals: {rising_beats}")

# ---------------------------------------------------------------------------
# The planner's private keys must not reach script.json
# ---------------------------------------------------------------------------
import new_video  # noqa: E402
import story  # noqa: E402

planned = beats.plan(topic="leak check", minutes=2)
blocks = [(f"headline {i}", f"narration for scene {i}") for i in range(1, len(planned["scenes"]) + 1)]
script, missing = new_video.build_script(
    planned, "A Title", blocks, voice="hm_omega", speed=0.94, captions={"enabled": True}, show_headline=True
)

assert not missing, f"scenes {missing} came back empty"
assert len(script["scenes"]) == len(planned["scenes"]), "scene count changed"
leaked = sorted(
    {key for scene in script["scenes"] for key in scene if key.startswith("_")}
)
assert not leaked, f"planner-private keys leaked into script.json: {leaked}"
assert script["scenes"][0]["headline"] == "headline 1", "headline not bound to scene 1"
assert script["scenes"][0]["lines"][0]["text"] == "narration for scene 1", "narration misbound"
print("planned scenes pass through wholesale, private keys stripped, narration bound by index.")

# ---------------------------------------------------------------------------
# The runtime contract is published once, and it tiles the scenes exactly
# ---------------------------------------------------------------------------
for minutes in (1, 5, 10, 30):
    plan = beats.plan(topic="runtime contract", minutes=minutes)
    blueprint = plan["blueprint"]
    scenes = plan["scenes"]

    assert blueprint["sceneCount"] == len(scenes), f"{minutes}m blueprint/plan scene mismatch"
    assert blueprint["sceneCount"] == plan["target"]["scenes"], f"{minutes}m blueprint disagrees with target"
    assert blueprint["totalWords"] == plan["target"]["totalWords"], f"{minutes}m blueprint word total drifted"
    tiled = [n for movement in blueprint["movements"] for n in movement["scenes"]]
    assert tiled == list(range(1, blueprint["sceneCount"] + 1)), (
        f"{minutes}m blueprint movements do not tile 1..{blueprint['sceneCount']}"
    )
    assert sum(m["sceneCount"] for m in blueprint["movements"]) == blueprint["sceneCount"]
    for movement in blueprint["movements"]:
        assert movement["range"] == [movement["scenes"][0], movement["scenes"][-1]]
print("every plan publishes a blueprint that tiles its scenes exactly once.")

# ---------------------------------------------------------------------------
# A scene is never called a "part", because that is how a story becomes a set
# ---------------------------------------------------------------------------
duties = [duty for _movement, _beat, duty in beats.story_plan(25)]
assert not any("Part " in duty for duty in duties), "a duty still calls a scene a 'Part'"
assert any("continue the same story" in duty for duty in duties), "duties lost their continuation wording"
print("scene duties say 'continue', never 'Part N of M'.")

# ---------------------------------------------------------------------------
# Chunks tile the story exactly, contiguously
# ---------------------------------------------------------------------------
for scene_count in (1, 5, 7, 8, 9, 24, 25, 100):
    chunks = story.plan_chunks(scene_count, 8)
    assert chunks[0][0] == 1, f"{scene_count} chunks do not start at 1"
    assert chunks[-1][1] == scene_count, f"{scene_count} chunks do not reach the end"
    for (first, last), (next_first, _next_last) in zip(chunks, chunks[1:]):
        assert last + 1 == next_first, f"{scene_count} chunks are not contiguous"
    covered = [n for first, last in chunks for n in range(first, last + 1)]
    assert covered == list(range(1, scene_count + 1)), f"{scene_count} chunks drop or duplicate a scene"
    assert all(last - first + 1 <= 8 for first, last in chunks), f"{scene_count} chunk is oversized"
assert story.plan_chunks(5, 99) == [(1, 5)], "a chunk size past the end must still be one chunk"
print("chunks tile every scene exactly once, contiguously, at every length.")

# ---------------------------------------------------------------------------
# A continuation call sees the bible, the tail verbatim, and where it stands
# ---------------------------------------------------------------------------
plan = beats.plan(topic="continuity", minutes=10)
bible = story.normalize_bible(
    {"logline": "एक खोया शहर और उसे खोजने वाला।", "arc": [{"movement": "hook", "scenes": [1]}]},
    plan,
)
chunks = story.plan_chunks(len(plan["scenes"]), 8)
written = [
    story.WrittenScene(
        n=n,
        headline=f"दृश्य {n}",
        visual="lantern",
        # Long enough that the summary line really does truncate it.
        body=" ".join([f"शब्द{n}"] + ["गहरा"] * 20),
    )
    for n in range(1, chunks[0][1] + 1)
]

first_prompt = story.build_chunk_prompt(plan, bible, chunks[0], [], is_hindi=True)
second_prompt = story.build_chunk_prompt(plan, bible, chunks[1], written, is_hindi=True)
last_prompt = story.build_chunk_prompt(plan, bible, chunks[-1], written, is_hindi=True)

assert "STORY BIBLE" in second_prompt, "continuation lost the bible"
assert f"scenes {chunks[1][0]}-{chunks[1][1]} of {len(plan['scenes'])}" in second_prompt, "wrong absolute range"
assert written[-1].body in second_prompt, "the previous scene is not shown verbatim"
assert written[-2].body in second_prompt, "the tail is thinner than two scenes"
assert written[0].body not in second_prompt, "the writer is being fed scenes it must not repeat"
assert "Do NOT restart" in second_prompt, "no restart instruction"
assert "Do NOT summarise and do NOT conclude" in second_prompt, "no continuation contract"
assert "open the story" in first_prompt, "the first chunk is not told to open the story"
assert "Do NOT restart" not in first_prompt, "the first chunk cannot continue from nothing"
assert "final chunk" in last_prompt, "the last chunk is not told to resolve the story"
assert f"scenes {chunks[-1][0]}-{chunks[-1][1]} out of {len(plan['scenes'])}" in last_prompt, (
    "the final chunk is not told where it sits in the story"
)
print("continuation prompts carry the bible, the verbatim tail and the absolute range.")

# ---------------------------------------------------------------------------
# Prose is bound by index: a dropped scene must not shift the ones after it
# ---------------------------------------------------------------------------
sample = "\n".join(
    [
        "TITLE: टेस्ट",
        "###1|पहला दृश्य | VISUAL: ancient corridor",
        "कहानी यहीं से शुरू होती है।",
        "###2|दूसरा दृश्य | VISUAL: stone gate",
        "दूसरा हिस्सा आगे बढ़ता है।",
        "###4|चौथा दृश्य | VISUAL: deep chamber",
        "चौथा हिस्सा कहानी पूरी करता है।",
    ]
)
title, parsed = story.parse_blocks(sample, 4)
assert [scene.n for scene in parsed] == [1, 2, 4], f"scene numbers not preserved: {[s.n for s in parsed]}"
assert title == "टेस्ट"

plan = beats.plan(topic="binding", minutes=1, scenes=4)
script, missing = new_video.build_script(
    plan, "T", parsed, voice="hm_omega", speed=1.0, captions={}
)
assert missing == [3], f"the dropped scene should be the only gap, got {missing}"
assert script["scenes"][2]["lines"] == [], "the gap must stay a gap"
assert script["scenes"][3]["lines"][0]["text"] == "चौथा हिस्सा कहानी पूरी करता है।", (
    "scene 4's prose drifted onto another scene"
)
print("a dropped scene leaves a gap instead of dragging every later scene out of place.")

# Unnumbered prose still binds in order, and the tail is simply reported missing.
legacy = [("पहला", "पहली पंक्ति"), ("दूसरा", "दूसरी पंक्ति")]
plan = beats.plan(topic="legacy binding", minutes=2)
script, missing = new_video.build_script(plan, "T", legacy, voice="hm_omega", speed=1.0, captions={})
assert script["scenes"][0]["lines"][0]["text"] == "पहली पंक्ति"
assert script["scenes"][1]["lines"][0]["text"] == "दूसरी पंक्ति"
assert missing == list(range(3, len(plan["scenes"]) + 1)), f"unexpected gaps: {missing}"
assert not any(key.startswith("_") for scene in script["scenes"] for key in scene)
print("unnumbered narration still binds positionally, gaps reported honestly.")

# ---------------------------------------------------------------------------
# Validation catches the things that make a story read as disconnected
# ---------------------------------------------------------------------------
plan = beats.plan(topic="validation", minutes=2)
duty_rows = story.scene_duties(plan)
good = [
    story.WrittenScene(
        n=row["n"],
        headline=f"दृश्य {row['n']}",
        # Distinct opening words per scene, exact word count, so nothing but the
        # injected fault should ever be reported.
        body=" ".join([f"प्रसंग{row['n']}"] * row["words"]),
    )
    for row in duty_rows
]
assert story.validate_script(good, plan) == [], "a clean script reported problems"

recapped = list(good)
recapped[2] = story.WrittenScene(n=3, headline="दृश्य 3", body="जैसा कि हमने पहले बताया, सब वैसा ही था।")
kinds = {issue.kind for issue in story.validate_script(recapped, plan)}
assert "restart" in kinds, f"a recap was not caught: {kinds}"

gapped = [scene for scene in good if scene.n != 2]
issues = story.validate_script(gapped, plan)
assert any(issue.kind == "missing" and issue.scene == 2 for issue in issues), "a missing scene was not caught"

starved = list(good)
starved[0] = story.WrittenScene(n=1, headline="दृश्य 1", body="बहुत छोटा।")
assert any(issue.kind == "length" for issue in story.validate_script(starved, plan)), "a starved scene was not caught"
print("validation catches recaps, gaps and starved scenes; a clean script stays quiet.")

# ---------------------------------------------------------------------------
# The bible carries the whole length, and degrades instead of exploding
# ---------------------------------------------------------------------------
plan = beats.plan(topic="bible", minutes=2)
expected_scenes = len(plan["scenes"])


def good_bible_llm(_prompt: str, **_kwargs) -> str:
    return json.dumps(
        {
            "logline": "एक राजा और उसका खोया शहर।",
            "arc": [{"movement": "hook", "scenes": [1], "summary": "शुरुआत"}],
            "sceneBeats": [
                {"n": n, "whatHappens": f"घटना {n}", "endsOn": f"सवाल {n}", "words": 40}
                for n in range(1, expected_scenes + 1)
            ],
            "payoff": {"scene": expected_scenes, "resolves": ["सवाल 1"]},
        },
        ensure_ascii=False,
    )


bible, report = story.build_bible(plan, good_bible_llm, attempts=1, verbose=False)
assert report["accepted"] is True, f"a good bible was rejected: {report['problems']}"
assert [entry["n"] for entry in bible["sceneBeats"]] == list(range(1, expected_scenes + 1)), (
    "the bible does not cover every planned scene"
)
assert bible["span"]["totalScenes"] == expected_scenes
assert bible["span"]["totalWords"] == plan["target"]["totalWords"], "the bible lost the word budget"
assert all(entry.get("words") for entry in bible["sceneBeats"])


def broken_llm(_prompt: str, **_kwargs) -> str:
    raise RuntimeError("503 service unavailable")


fallback, report = story.build_bible(plan, broken_llm, attempts=2, verbose=False)
assert fallback is not None, "a broken writer must degrade, not return nothing"
assert report["fallback"] is True
assert len(fallback["sceneBeats"]) == expected_scenes, "the fallback bible does not cover every scene"
assert report["attempts"], "failures were not recorded"

assert story.parse_bible('```json\n{"logline": "x"}\n```')["logline"] == "x"
assert story.parse_bible('Here you go:\n{"logline": "y"}\nHope that helps!')["logline"] == "y"
assert story.parse_bible("no json at all") is None
assert story.bible_ok({"logline": "x"}, plan), "a bible with no arc or scenes must be rejected"
print(f"the bible spans all {expected_scenes} scenes and survives a dead writer.")

print("\nStory planner verified: complete arc at every length, exact allocation,")
print("distinct per-scene instructions, nothing private leaked into script.json,")
print("and one continuous story told in order across chunked calls.")
