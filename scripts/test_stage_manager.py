"""
Unit tests for the stage manager and checkpointing system in tube2dscript.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from stage_manager import (
    CANONICAL_STAGES,
    STAGE_IMAGES,
    STAGE_RENDER,
    STAGE_SCRIPT_SEO,
    STAGE_VOICE,
    PipelineStateManager,
    normalize_stage,
    stage_index,
)


class TestStageManager(unittest.TestCase):
    def test_normalize_stage(self):
        self.assertEqual(normalize_stage("1"), STAGE_SCRIPT_SEO)
        self.assertEqual(normalize_stage("script"), STAGE_SCRIPT_SEO)
        self.assertEqual(normalize_stage("seo"), STAGE_SCRIPT_SEO)
        self.assertEqual(normalize_stage("stage1"), STAGE_SCRIPT_SEO)
        self.assertEqual(normalize_stage("script_seo"), STAGE_SCRIPT_SEO)

        self.assertEqual(normalize_stage("2"), STAGE_VOICE)
        self.assertEqual(normalize_stage("voice"), STAGE_VOICE)
        self.assertEqual(normalize_stage("tts"), STAGE_VOICE)
        self.assertEqual(normalize_stage("audio"), STAGE_VOICE)

        self.assertEqual(normalize_stage("3"), STAGE_IMAGES)
        self.assertEqual(normalize_stage("images"), STAGE_IMAGES)
        self.assertEqual(normalize_stage("image"), STAGE_IMAGES)
        self.assertEqual(normalize_stage("visual"), STAGE_IMAGES)

        self.assertEqual(normalize_stage("4"), STAGE_RENDER)
        self.assertEqual(normalize_stage("render"), STAGE_RENDER)
        self.assertEqual(normalize_stage("export"), STAGE_RENDER)

        self.assertIsNone(normalize_stage("invalid_stage"))
        self.assertIsNone(normalize_stage(None))

    def test_stage_index(self):
        self.assertEqual(stage_index(STAGE_SCRIPT_SEO), 1)
        self.assertEqual(stage_index(STAGE_VOICE), 2)
        self.assertEqual(stage_index(STAGE_IMAGES), 3)
        self.assertEqual(stage_index(STAGE_RENDER), 4)

    def test_should_execute_stage_filtering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            mgr = PipelineStateManager(state_file)

            # Explicit stage mode (--stage voice)
            self.assertFalse(mgr.should_execute_stage(STAGE_SCRIPT_SEO, explicit_stage="voice"))
            self.assertTrue(mgr.should_execute_stage(STAGE_VOICE, explicit_stage="voice"))
            self.assertFalse(mgr.should_execute_stage(STAGE_IMAGES, explicit_stage="voice"))
            self.assertFalse(mgr.should_execute_stage(STAGE_RENDER, explicit_stage="voice"))

            # Bounded from_stage (--from-stage images)
            self.assertFalse(mgr.should_execute_stage(STAGE_SCRIPT_SEO, from_stage="images"))
            self.assertFalse(mgr.should_execute_stage(STAGE_VOICE, from_stage="images"))
            self.assertTrue(mgr.should_execute_stage(STAGE_IMAGES, from_stage="images"))
            self.assertTrue(mgr.should_execute_stage(STAGE_RENDER, from_stage="images"))

            # Bounded to_stage (--to-stage voice)
            self.assertTrue(mgr.should_execute_stage(STAGE_SCRIPT_SEO, to_stage="voice"))
            self.assertTrue(mgr.should_execute_stage(STAGE_VOICE, to_stage="voice"))
            self.assertFalse(mgr.should_execute_stage(STAGE_IMAGES, to_stage="voice"))
            self.assertFalse(mgr.should_execute_stage(STAGE_RENDER, to_stage="voice"))

    def test_checkpoint_recording_and_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            mgr = PipelineStateManager(state_file)

            # Initially no stages completed
            self.assertFalse(mgr.is_stage_completed(STAGE_SCRIPT_SEO))
            self.assertTrue(mgr.should_execute_stage(STAGE_SCRIPT_SEO, resume=True))

            # Complete stage 1
            mgr.record_stage_start(STAGE_SCRIPT_SEO)
            mgr.record_stage_complete(STAGE_SCRIPT_SEO, artifacts=["out/script.json"])
            self.assertTrue(mgr.is_stage_completed(STAGE_SCRIPT_SEO))

            # Reload manager from disk
            mgr2 = PipelineStateManager(state_file)
            self.assertTrue(mgr2.is_stage_completed(STAGE_SCRIPT_SEO))
            self.assertIn("out/script.json", mgr2.data["stages"][STAGE_SCRIPT_SEO]["artifacts"])

            # On resume, stage 1 is skipped, stage 2 runs
            self.assertFalse(mgr2.should_execute_stage(STAGE_SCRIPT_SEO, resume=True))
            self.assertTrue(mgr2.should_execute_stage(STAGE_VOICE, resume=True))

    def test_pause_after_triggers_callback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            mgr = PipelineStateManager(state_file)

            mock_pause_cb = MagicMock()
            # If pause_after == voice and current is script_seo -> continues (returns True)
            cont = mgr.prompt_after_stage(STAGE_SCRIPT_SEO, pause_after="voice", on_pause=mock_pause_cb)
            self.assertTrue(cont)
            mock_pause_cb.assert_not_called()

            # If pause_after == script_seo -> triggers pause callback & halts (returns False)
            cont2 = mgr.prompt_after_stage(STAGE_SCRIPT_SEO, pause_after="script_seo", on_pause=mock_pause_cb)
            self.assertFalse(cont2)
            mock_pause_cb.assert_called_once()


if __name__ == "__main__":
    unittest.main()
