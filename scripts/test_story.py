"""Verify the story planner always produces a complete arc.

Beats are visual grammar, not a story. Cycling them is how a long video used to
loop the same six shots and never actually end, so these invariants are the thing
that stops a video coming out half-finished at any length.

Run with:  .venv\\Scripts\\python.exe scripts\\test_story.py
"""

from __future__ import annotations

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
    assert key in beats.MOVEMENT_SETS[-1][1], f"{key} is missing from the shortest set"

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
    assert ESSENTIAL <= set(order), f"{minutes} min is missing {ESSENTIAL - set(order)}"
    assert order[0] in OPENERS, f"{minutes} min opens on {order[0]}"
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

planned = beats.plan(topic="leak check", minutes=2)
blocks = [(f"headline {i}", f"narration for scene {i}") for i in range(1, len(planned["scenes"]) + 1)]
script, missing = new_video.build_script(
    planned, "A Title", blocks, voice="hm_omega", speed=0.94, captions={"enabled": True}
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

print("\nStory planner verified: complete arc at every length, exact allocation,")
print("distinct per-scene instructions, nothing private leaked into script.json.")
