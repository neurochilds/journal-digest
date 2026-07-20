import json
import sys
import tempfile
import types
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

# Keep unit tests deterministic and runnable before optional network clients are
# installed. Production and GitHub Actions install requirements.txt normally.
try:
    import feedparser  # noqa: F401
except ModuleNotFoundError:
    sys.modules["feedparser"] = types.SimpleNamespace(parse=Mock())

try:
    import requests  # noqa: F401
except ModuleNotFoundError:
    requests_stub = types.ModuleType("requests")
    requests_stub.get = Mock()
    requests_stub.exceptions = types.SimpleNamespace(Timeout=TimeoutError)
    sys.modules["requests"] = requests_stub

try:
    import openai  # noqa: F401
except ModuleNotFoundError:
    openai_stub = types.ModuleType("openai")
    openai_stub.OpenAI = Mock
    sys.modules["openai"] = openai_stub

import paper_tracker


class TitleMatchingTests(unittest.TestCase):
    def test_accepts_punctuation_and_subtitle_variation(self):
        self.assertTrue(
            paper_tracker.title_matches(
                "Latent states in CA1: a multisensory study",
                "Latent states in CA1 — a multisensory study",
            )
        )

    def test_rejects_first_result_from_unrelated_paper(self):
        self.assertFalse(
            paper_tracker.title_matches(
                "Hippocampal coding during audiovisual conflict",
                "Visual coding in primary sensory cortex",
            )
        )

    @patch("paper_tracker.requests.get")
    def test_crossref_doi_requires_matching_title(self, mock_get):
        response = Mock(status_code=200)
        response.json.return_value = {
            "message": {"items": [{"title": ["Unrelated result"], "DOI": "10.1/wrong"}]}
        }
        mock_get.return_value = response
        self.assertEqual(paper_tracker.fetch_doi_from_crossref("Target paper title"), "")


class RetrievalTests(unittest.TestCase):
    @patch("paper_tracker.requests.get")
    def test_openalex_http_error_is_not_an_empty_success(self, mock_get):
        mock_get.return_value = Mock(status_code=503)
        with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
            paper_tracker.fetch_papers_from_openalex("2026-07-19", "2026-07-20")


class StateTests(unittest.TestCase):
    def test_atomic_state_save_preserves_existing_and_adds_new(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "seen.json"
            state.write_text(json.dumps({"old": "2026-07-20T00:00:00"}), encoding="utf-8")
            with patch.object(paper_tracker, "SEEN_PAPERS_FILE", state):
                paper_tracker.save_seen_papers({"new"})
            saved = json.loads(state.read_text(encoding="utf-8"))
            self.assertIn("old", saved)
            self.assertIn("new", saved)
            self.assertFalse(list(state.parent.glob(".seen.json.*.tmp")))

    def test_mark_papers_seen_writes_both_stable_ids(self):
        paper = {
            "title": "A paper",
            "link": "https://doi.org/10.1/example",
            "date": datetime(2026, 7, 20),
        }
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "seen.json"
            with patch.object(paper_tracker, "SEEN_PAPERS_FILE", state):
                paper_tracker.mark_papers_seen(set(), [paper])
            saved = json.loads(state.read_text(encoding="utf-8"))
            self.assertIn(paper_tracker.get_paper_id(paper), saved)
            self.assertIn(paper_tracker.get_title_id(paper), saved)


class ScoringTests(unittest.TestCase):
    def test_model_failure_is_not_silently_converted_to_zero(self):
        client = Mock()
        client.chat.completions.create.side_effect = RuntimeError("API unavailable")
        with self.assertRaisesRegex(RuntimeError, "LLM scoring failed"):
            paper_tracker.get_llm_relevance_score(
                client, {"title": "Example", "abstract": "A" * 100}
            )

    def test_valid_model_json_is_accepted(self):
        message = Mock(content='{"score": 77, "reason": "Directly tests the target mechanism."}')
        client = Mock()
        client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=message)]
        )
        self.assertEqual(
            paper_tracker.get_llm_relevance_score(
                client, {"title": "Example", "abstract": "A" * 100}
            ),
            (77, "Directly tests the target mechanism."),
        )


class DeliveryTransactionTests(unittest.TestCase):
    def _run_with_email(self, email_side_effect=None):
        artifact_dir = Path(paper_tracker.__file__).resolve().parent
        backup_files_before = set(artifact_dir.glob("digest_*.txt"))
        paper = {
            "title": "Hippocampal latent state coding",
            "link": "https://doi.org/10.1/example",
            "abstract": "hippocampal task state " * 20,
            "journal": "Example Journal",
            "date": datetime(2026, 7, 20),
            "authors": "A. Researcher",
        }
        patches = [
            patch.object(paper_tracker, "OPENAI_API_KEY", "test"),
            patch.object(paper_tracker, "GMAIL_ADDRESS", "from@example.org"),
            patch.object(paper_tracker, "GMAIL_APP_PASSWORD", "test"),
            patch.object(paper_tracker, "RECIPIENT_EMAIL", "to@example.org"),
            patch.object(paper_tracker, "OpenAI", return_value=Mock()),
            patch.object(paper_tracker, "load_seen_papers", return_value=set()),
            patch.object(paper_tracker, "fetch_papers_from_openalex", return_value=[paper]),
            patch.object(
                paper_tracker,
                "get_llm_relevance_score",
                return_value=(90, "Directly relevant."),
            ),
            patch.object(paper_tracker, "summarize_paper", return_value="Faithful summary."),
            patch.object(paper_tracker, "send_email", side_effect=email_side_effect),
            patch.object(paper_tracker, "mark_papers_seen"),
        ]
        entered = [item.start() for item in patches]
        try:
            exit_code = paper_tracker.main(
                start_date="2026-07-20", end_date="2026-07-20", historical=True
            )
            return exit_code, entered[-1]
        finally:
            for item in reversed(patches):
                item.stop()
            for backup in set(artifact_dir.glob("digest_*.txt")) - backup_files_before:
                backup.unlink()

    def test_email_failure_does_not_consume_paper(self):
        exit_code, mark_mock = self._run_with_email(RuntimeError("SMTP unavailable"))
        self.assertEqual(exit_code, 1)
        mark_mock.assert_not_called()

    def test_email_success_commits_processed_paper(self):
        exit_code, mark_mock = self._run_with_email()
        self.assertEqual(exit_code, 0)
        mark_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
