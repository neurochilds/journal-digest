import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config


class ProfileOverrideTests(unittest.TestCase):
    def tearDown(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PAPER_TRACKER_PROFILE", None)
            importlib.reload(config)

    def test_json_profile_overrides_profile_and_keywords(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(
                json.dumps(
                    {
                        "research_profile": "Lab-specific profile",
                        "keywords": {"topic": ["hippocampal inference"]},
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"PAPER_TRACKER_PROFILE": str(path)}):
                reloaded = importlib.reload(config)
                self.assertEqual(reloaded.RESEARCH_PROFILE, "Lab-specific profile")
                self.assertEqual(reloaded.KEYWORDS, {"topic": ["hippocampal inference"]})

    def test_missing_profile_fails_loudly(self):
        with patch.dict(os.environ, {"PAPER_TRACKER_PROFILE": "/definitely/missing.json"}):
            with self.assertRaisesRegex(RuntimeError, "does not exist"):
                importlib.reload(config)
