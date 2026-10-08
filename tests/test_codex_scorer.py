import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from codex_scorer import CodexScorer
from relevance import SCORE_FIELDS


class CodexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        (self.home / 'auth.json').write_text(json.dumps({'auth_mode': 'chatgpt', 'tokens': {'test': True}}))
        self.env = patch.dict(os.environ, {'PAPER_SCOUT_CODEX_HOME': str(self.home),
                              'PAPER_SCOUT_CODEX_BIN': '/test/codex',
                              'PAPER_SCOUT_CODEX_MODEL': 'gpt-6.1-sol',
                              'OPENAI_API_KEY': 'api-secret', 'GMAIL_APP_PASSWORD': 'smtp-secret'})
        self.env.start(); self.addCleanup(self.env.stop)
        self.client = CodexScorer()
        self.paper = {'title': 'Test', 'abstract': 'x' * 2400}

    def request(self, rows, event=None, returncode=0):
        def start(command, **kwargs):
            process = Mock(returncode=returncode)
            def communicate(prompt, timeout):
                self.prompt, self.child_env, self.command = prompt.decode(), kwargs['env'], command
                Path(command[command.index('-o') + 1]).write_text(json.dumps({'results': rows}))
                kwargs['stdout'].write((json.dumps(event or {'type': 'turn.completed'}) + '\n').encode())
            process.communicate.side_effect = communicate
            return process
        with patch('codex_scorer.subprocess.Popen', side_effect=start):
            return self.client.request('Rubric', [self.paper], {'score': {'type': 'integer'},
                                                             'reason': {'type': 'string'}})

    def test_subscription_only_isolated_invocation(self):
        rows = self.request([{'id': 0, 'score': 75, 'reason': 'Relevant'}])
        self.assertEqual(rows[0]['score'], 75)
        self.assertNotIn('OPENAI_API_KEY', self.child_env)
        self.assertNotIn('GMAIL_APP_PASSWORD', self.child_env)
        self.assertIn('forced_login_method="chatgpt"', self.command)
        self.assertIn('read-only', self.command)
        self.assertIn('shell_tool', self.command)
        self.assertIn('untrusted data', self.prompt)
        self.assertIn('x' * 2400, self.prompt)

    def test_api_auth_is_rejected_before_execution(self):
        (self.home / 'auth.json').write_text(json.dumps({'auth_mode': 'apikey', 'OPENAI_API_KEY': 'test'}))
        with self.assertRaisesRegex(RuntimeError, 'ChatGPT authentication'):
            CodexScorer()

    def test_missing_auth_stops_before_execution(self):
        (self.home / 'auth.json').unlink()
        with self.assertRaises(FileNotFoundError):
            CodexScorer()

    def test_rejects_missing_duplicate_and_mismatched_ids(self):
        for rows in ([], [{'id': 1, 'score': 75, 'reason': 'R'}],
                     [{'id': True, 'score': 75, 'reason': 'R'}],
                     [{'id': 0, 'score': 75, 'reason': 'R'}] * 2):
            with self.subTest(rows=rows), self.assertRaises(RuntimeError):
                self.request(rows)

    def test_rejects_invalid_scores_and_text(self):
        for score, reason in ((101, 'R'), (-1, 'R'), (True, 'R'), (75, ''), (75, 'x' * 2001)):
            with self.subTest(score=score), self.assertRaises(RuntimeError):
                self.request([{'id': 0, 'score': score, 'reason': reason}])

    def test_tool_events_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, 'tool activity'):
            self.request([{'id': 0, 'score': 75, 'reason': 'R'}],
                         {'type': 'item.completed', 'item': {'type': 'command_execution'}})

    def test_disabled_code_host_warning_is_not_a_tool_call(self):
        rows = self.request([{'id': 0, 'score': 75, 'reason': 'R'}],
                            {'type': 'item.completed', 'item': {'type': 'error', 'message':
                             'Code Mode is unavailable because code-mode host is disabled. '
                             'Code mode will fail closed; enable `features.code_mode_host` '
                             'and install `codex-code-mode-host`.'}})
        self.assertEqual(rows[0]['score'], 75)

    def test_other_error_events_are_rejected(self):
        with self.assertRaises(RuntimeError):
            self.request([{'id': 0, 'score': 75, 'reason': 'R'}],
                         {'type': 'item.completed', 'item': {'type': 'error', 'message': 'Other failure'}})

    def test_provider_failure_does_not_return_partial_or_fall_back(self):
        with self.assertRaisesRegex(RuntimeError, 'account access or usage limits'):
            self.request([{'id': 0, 'score': 75, 'reason': 'R'}], returncode=1)

    def test_shared_deadline_stops_new_calls(self):
        self.client.deadline = 0
        with patch('codex_scorer.subprocess.Popen') as start:
            with self.assertRaisesRegex(RuntimeError, 'time budget'):
                self.request([])
            start.assert_not_called()

    def test_batches_preserve_paper_mapping_and_complete_abstract(self):
        papers = [dict(self.paper, title=str(i)) for i in range(23)]
        def request(instructions, batch, fields):
            self.assertTrue(all(len(p['abstract']) == 2400 for p in batch))
            return [{'score': 70, 'reason': p['title'], 'limitation': 'No direct test.', 'kind': 'paper'}
                    for p in batch]
        with patch.object(self.client, 'request', side_effect=request) as ask:
            self.client.score_many(papers, 'Rubric')
        self.assertEqual([len(c.args[1]) for c in ask.call_args_list], [10, 10, 3])
        self.assertEqual(self.client.scores[self.client.key(papers[-1])], (70, '22'))

    def test_missing_abstract_is_not_summarized(self):
        with patch.object(self.client, 'request') as ask:
            self.client.summarize_many([{'title': 'Missing', 'abstract': ''}])
            ask.assert_not_called()

    def test_title_only_cannot_be_presented_as_direct_evidence(self):
        paper = {'title': 'Auditory hippocampal navigation'}
        with patch.object(self.client, 'request', return_value=[
                {'score': 99, 'reason': 'Promising title.', 'limitation': 'No abstract.', 'kind': 'paper'}]):
            self.client.score_many([paper], 'Rubric')
        self.assertEqual(paper['llm_score'], 69)

    def test_successful_batches_checkpoint_before_later_failure(self):
        papers = [dict(self.paper, title=str(i)) for i in range(11)]
        checkpoint = Mock()
        with patch.object(self.client, 'request', side_effect=[
                [{'score': 70, 'reason': 'Relevant', 'limitation': 'No direct test.', 'kind': 'paper'}
                 for _ in range(10)], RuntimeError('limit')]):
            with self.assertRaisesRegex(RuntimeError, 'limit'):
                self.client.score_many(papers, 'Rubric', on_batch=checkpoint)
        checkpoint.assert_called_once()
        self.assertTrue(all(p.get('llm_score') == 70 for p in papers[:10]))
        self.assertNotIn('llm_score', papers[-1])

    def test_score_and_summary_receive_identical_late_results(self):
        paper = {'title': 'Rule selection', 'abstract': 'Background. ' * 180 + 'RESULT: silencing changed rule bias.'}
        captured = []
        def start(command, **kwargs):
            process = Mock(returncode=0)
            def communicate(prompt, timeout):
                payload = json.loads(prompt.decode().rsplit('\n', 1)[-1])
                captured.append(payload[0])
                fields = json.loads(Path(command[command.index('--output-schema') + 1]).read_text())['properties']['results']['items']['properties']
                row = ({'id': 0, 'score': 82, 'reason': 'Rule versus action control.',
                        'limitation': 'No hippocampal recording.', 'kind': 'paper'} if 'score' in fields
                       else {'id': 0, 'summary': 'Silencing changed rule bias.'})
                Path(command[command.index('-o') + 1]).write_text(json.dumps({'results': [row]}))
                kwargs['stdout'].write(b'{"type":"turn.completed"}\n')
            process.communicate.side_effect = communicate
            return process
        with patch('codex_scorer.subprocess.Popen', side_effect=start):
            self.client.score_many([paper], 'Rubric')
            self.client.summarize_many([paper])
        self.assertEqual(captured[0], captured[1])
        self.assertTrue(captured[0]['abstract'].endswith('RESULT: silencing changed rule bias.'))

    def test_invalid_resource_category_is_rejected(self):
        def start(command, **kwargs):
            process = Mock(returncode=0)
            def communicate(prompt, timeout):
                Path(command[command.index('-o') + 1]).write_text(json.dumps({'results': [
                    {'id': 0, 'score': 74, 'reason': 'Data.', 'limitation': 'No results.', 'kind': 'unknown'}]}))
                kwargs['stdout'].write(b'{"type":"turn.completed"}\n')
            process.communicate.side_effect = communicate
            return process
        with patch('codex_scorer.subprocess.Popen', side_effect=start):
            with self.assertRaisesRegex(RuntimeError, 'category'):
                self.client.request('Rubric', [self.paper], SCORE_FIELDS)


if __name__ == '__main__':
    unittest.main()
