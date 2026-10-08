import copy
import io
import json
import tempfile
import time
import unittest
from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import requests

import openalex_client as api
import paper_tracker as tracker


def response(status=200, data=None, headers=None):
    result = Mock(status_code=status, headers=headers or {})
    result.json.return_value = {} if data is None else data
    return result


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.get = self.stack.enter_context(patch.object(api.requests, 'get'))
        self.sleep = self.stack.enter_context(patch.object(api.time, 'sleep'))
        self.stack.enter_context(redirect_stdout(io.StringIO()))

    def test_paid_429_stops_immediately(self):
        self.get.return_value = response(429, {'error': 'Plan upgrade required', 'message': 'Requires a Premium plan'})
        with self.assertRaisesRegex(RuntimeError, 'paid plan'):
            api.get_json('https://api.openalex.org/works')
        self.assertEqual(self.get.call_count, 1)
        self.sleep.assert_not_called()

    def test_transient_429_honours_retry_hint(self):
        self.get.side_effect = [response(429, headers={'Retry-After': '3'}), response(data={'ok': True})]
        self.assertEqual(api.get_json('https://api.openalex.org/works'), {'ok': True})
        self.sleep.assert_called_once_with(3)

    def test_repeated_429_is_bounded(self):
        self.get.return_value = response(429)
        with self.assertRaisesRegex(RuntimeError, 'exhausted 5 attempts'):
            api.get_json('https://api.openalex.org/works')
        self.assertEqual(self.get.call_count, 5)
        self.assertEqual(self.sleep.call_count, 4)

    def test_timeouts_are_bounded(self):
        self.get.side_effect = requests.exceptions.Timeout('sensitive upstream details')
        with self.assertRaisesRegex(RuntimeError, 'network failure') as error:
            api.get_json('https://api.openalex.org/works')
        self.assertNotIn('sensitive', str(error.exception))
        self.assertEqual(self.get.call_count, 5)

    def test_daily_budget_is_not_retried(self):
        self.get.return_value = response(429, {'error': 'Daily budget exceeded'})
        with self.assertRaisesRegex(RuntimeError, 'daily budget'):
            api.get_json('https://api.openalex.org/works')
        self.assertEqual(self.get.call_count, 1)

    def test_long_server_wait_fails_instead_of_sleeping(self):
        self.get.return_value = response(429, headers={'Retry-After': '3600'})
        with self.assertRaisesRegex(RuntimeError, 'retry delay'):
            api.get_json('https://api.openalex.org/works')
        self.sleep.assert_not_called()

    def test_expired_budget_makes_no_request(self):
        with self.assertRaisesRegex(RuntimeError, 'time budget'):
            api.get_json('https://api.openalex.org/works', deadline=time.monotonic() - 1)
        self.get.assert_not_called()

    def test_authentication_uses_header_not_url(self):
        self.get.return_value = response(data={'ok': True})
        with patch.object(api, 'OPENALEX_API_KEY', 'dummy-secret'):
            api.get_json('https://api.openalex.org/works', params={'per-page': 100})
        call = self.get.call_args
        self.assertNotIn('dummy-secret', call.args[0])
        self.assertNotIn('api_key', call.kwargs['params'])
        self.assertEqual(call.kwargs['headers']['Authorization'], 'Bearer dummy-secret')

    def test_non_retryable_error_fails_immediately(self):
        self.get.return_value = response(401)
        with self.assertRaisesRegex(RuntimeError, 'HTTP 401'):
            api.get_json('https://api.openalex.org/works')
        self.assertEqual(self.get.call_count, 1)


class PaginationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.get = self.stack.enter_context(patch.object(tracker, 'openalex_get_json'))
        self.stack.enter_context(patch.object(tracker.time, 'sleep'))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.work = {'id': 'https://openalex.org/W1', 'title': 'Hippocampal navigation', 'publication_date': '2026-09-24'}

    def test_partial_page_failure_never_returns_partial_success(self):
        self.get.side_effect = [
            {'results': [self.work], 'meta': {'next_cursor': 'next'}},
            RuntimeError('OpenAlex HTTP 503'),
        ]
        with self.assertRaisesRegex(RuntimeError, 'HTTP 503'):
            tracker.fetch_papers_from_openalex('2026-09-01', '2026-10-03')

    def test_pagination_completes(self):
        self.get.side_effect = [
            {'results': [self.work], 'meta': {'next_cursor': 'next'}},
            {'results': [dict(self.work, id='https://openalex.org/W2')], 'meta': {'next_cursor': None}},
        ]
        papers = tracker.fetch_papers_from_openalex('2026-09-01', '2026-10-03')
        self.assertEqual(len(papers), 2)
        self.assertEqual(self.get.call_args.kwargs['params']['per-page'], 100)
        self.assertEqual(self.get.call_args.kwargs['params']['cursor'], 'next')

    def test_dataset_type_survives_retrieval(self):
        self.get.return_value = {'results': [dict(self.work, type='dataset')], 'meta': {'next_cursor': None}}
        papers = tracker.fetch_papers_from_openalex('2026-09-01', '2026-10-03')
        self.assertEqual(papers[0]['work_type'], 'dataset')
        self.assertIn('type', self.get.call_args.kwargs['params']['select'].split(','))

    def test_empty_terminal_page_is_valid(self):
        self.get.return_value = {'results': [], 'meta': {'next_cursor': None}}
        self.assertEqual(tracker.fetch_papers_from_openalex('2026-09-01', '2026-10-03'), [])

    def test_repeated_cursor_fails(self):
        self.get.return_value = {'results': [self.work], 'meta': {'next_cursor': '*'}}
        with self.assertRaisesRegex(RuntimeError, 'repeated'):
            tracker.fetch_papers_from_openalex('2026-09-01', '2026-10-03')

    def test_page_limit_fails_closed(self):
        self.get.side_effect = [
            {'results': [self.work], 'meta': {'next_cursor': str(i)}} for i in range(300)
        ]
        with self.assertRaisesRegex(RuntimeError, '300-page limit'):
            tracker.fetch_papers_from_openalex('2026-09-01', '2026-10-03')

    def test_busy_window_completes_past_previous_page_limit(self):
        self.get.side_effect = [
            {'results': [self.work], 'meta': {'next_cursor': str(i)}} for i in range(120)
        ] + [{'results': [self.work], 'meta': {'next_cursor': None}}]
        papers = tracker.fetch_papers_from_openalex('2026-08-04', '2026-10-03')
        self.assertEqual(len(papers), 121)
        self.assertEqual(self.get.call_count, 121)
        deadlines = {call.kwargs['deadline'] for call in self.get.call_args_list}
        self.assertEqual(len(deadlines), 1)

    def test_missing_pagination_metadata_fails(self):
        self.get.return_value = {'results': [self.work], 'meta': {}}
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            tracker.fetch_papers_from_openalex('2026-09-01', '2026-10-03')


class RunTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.paths = {}
        for setting, filename in [('SEEN_PAPERS_FILE', 'seen.json'), ('FIRST_OBSERVED_FILE', 'observed.json'),
                                  ('PENDING_PAPERS_FILE', 'pending.json'), ('DIGEST_LOG_FILE', 'log.csv')]:
            self.paths[setting] = self.directory / filename
            self.stack.enter_context(patch.object(tracker, setting, self.paths[setting]))
        self.stack.enter_context(patch.object(tracker, 'OPENAI_API_KEY', 'test-only'))
        for setting in ['GMAIL_ADDRESS', 'GMAIL_APP_PASSWORD', 'RECIPIENT_EMAIL']:
            self.stack.enter_context(patch.object(tracker, setting, 'test-only'))
        self.client = self.stack.enter_context(patch.object(tracker, 'OpenAI'))
        self.fetch = self.stack.enter_context(patch.object(tracker, 'fetch_papers_from_openalex'))
        self.score = self.stack.enter_context(patch.object(tracker, 'get_llm_relevance_score', return_value=(95, 'Relevant.')))
        def score(client, paper):
            paper['llm_limitation'], paper['llm_kind'] = 'Abstract only.', 'paper'
            return self.score.return_value
        self.score.side_effect = score
        self.stack.enter_context(patch.object(tracker, 'summarize_paper', return_value='Summary.'))
        self.send = self.stack.enter_context(patch.object(tracker, 'send_email'))
        self.stack.enter_context(patch.object(tracker.time, 'sleep'))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.paper = {'title': 'Hippocampal navigation', 'link': 'https://doi.org/10.1/example',
                      'abstract': 'hippocampal memory and task state ' * 10, 'journal': 'Example Journal',
                      'date': datetime.now(), 'authors': 'Example', 'openalex_id': 'https://openalex.org/W1'}
        self.fetch.side_effect = lambda *args, **kwargs: [copy.deepcopy(self.paper)]

    def test_default_uses_one_free_publication_overlap(self):
        self.assertEqual(tracker.main(fetch_only=True), 0)
        self.fetch.assert_called_once_with((datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))

    def test_email_score_is_ai_priority_without_keyword_penalty(self):
        self.score.return_value = (93, 'Internal task representation modulated by vision.')
        tracker.main()
        html, text = self.send.call_args.args[1:]
        self.assertIn('93/100', html)
        self.assertIn('93/100', text)
        self.assertNotIn('Keyword:', html)
        self.assertIn('Direct relevance', html)

    def test_background_overflow_is_deferred_without_marking_seen(self):
        self.score.return_value = (62, 'Useful background.')
        papers = [dict(self.paper, title=f'Hippocampal background {i}', link=f'https://doi.org/10.1/{i}')
                  for i in range(5)]
        self.fetch.side_effect = lambda *a, **kw: copy.deepcopy(papers)
        tracker.main()
        pending = tracker.load_pending_papers()
        self.assertEqual(len(pending), 3)
        seen = tracker.load_seen_papers()
        self.assertTrue(all(tracker.get_paper_id(p) not in seen for p in pending))

    def test_explicit_window_is_preserved(self):
        tracker.main(start_date='2026-08-28', end_date='2026-10-03', fetch_only=True)
        self.fetch.assert_called_once_with('2026-08-28', '2026-10-03')

    def test_fetch_only_needs_no_ai_or_mail_credentials(self):
        with patch.object(tracker, 'OPENAI_API_KEY', ''), patch.object(tracker, 'GMAIL_ADDRESS', ''):
            self.assertEqual(tracker.main(fetch_only=True, preview_file=self.directory / 'preview.html'), 0)
        self.client.assert_not_called()
        self.score.assert_not_called()
        self.send.assert_not_called()
        self.assertTrue((self.directory / 'preview.html').exists())
        self.assertFalse(any(p.exists() for p in self.paths.values()))

    def test_dry_run_preserves_all_state_and_logs_byte_for_byte(self):
        initial = {'SEEN_PAPERS_FILE': '{}\n', 'FIRST_OBSERVED_FILE': '{}\n',
                   'PENDING_PAPERS_FILE': '[]\n', 'DIGEST_LOG_FILE': 'existing log\n'}
        for setting, text in initial.items():
            self.paths[setting].write_text(text)
        self.assertEqual(tracker.main(dry_run=True), 0)
        self.send.assert_not_called()
        for setting, text in initial.items():
            self.assertEqual(self.paths[setting].read_text(), text)

    def test_smtp_failure_queues_paper_and_returns_failure(self):
        self.send.side_effect = RuntimeError('SMTP unavailable')
        self.assertEqual(tracker.main(), 1)
        self.assertFalse(self.paths['SEEN_PAPERS_FILE'].exists())
        pending = json.loads(self.paths['PENDING_PAPERS_FILE'].read_text())
        self.assertEqual(pending[0]['llm_score'], 95)

    def test_retry_reuses_scores_and_commits_only_after_delivery(self):
        self.send.side_effect = [RuntimeError('SMTP unavailable'), None]
        self.assertEqual(tracker.main(), 1)
        self.assertEqual(tracker.main(), 0)
        self.assertEqual(self.score.call_count, 1)
        seen = json.loads(self.paths['SEEN_PAPERS_FILE'].read_text())
        self.assertIn(tracker.get_paper_id(self.paper), seen)
        self.assertIn(tracker.get_title_id(self.paper), seen)
        self.assertEqual(json.loads(self.paths['PENDING_PAPERS_FILE'].read_text()), [])

    def test_changed_rubric_or_model_rescreens_pending_scores(self):
        for changes in ({'llm_rubric': 'old-rubric'}, {'llm_model': 'old-model'}):
            with self.subTest(changes=changes):
                paper = dict(self.paper, llm_score=91, llm_reason='Old relevance.',
                             llm_model='gpt-5.1', llm_rubric=tracker.RELEVANCE_VERSION,
                             llm_input=tracker.score_input(self.paper))
                paper.update(changes)
                tracker.save_pending_papers([paper])
                self.score.reset_mock(); self.send.reset_mock()
                self.score.return_value = (10, 'Peripheral background.')
                self.assertEqual(tracker.main(dry_run=True), 0)
                self.score.assert_called_once()
                self.send.assert_not_called()
                self.assertEqual(tracker.load_pending_papers()[0]['llm_score'], 91)

    def test_auditory_navigation_survives_cap_ahead_of_generic_hippocampus(self):
        auditory = dict(self.paper, title='Auditory-guided navigation',
                        link='https://doi.org/10.1/auditory',
                        abstract='Mice used acoustic cues to navigate to a reward location.')
        self.fetch.side_effect = lambda *a, **kw: [copy.deepcopy(self.paper), copy.deepcopy(auditory)]
        self.assertEqual(tracker.main(max_llm_candidates=1, dry_run=True), 0)
        self.assertEqual(self.score.call_args.args[1]['title'], auditory['title'])

    def test_digest_overflow_remains_queued_and_unseen(self):
        second = dict(self.paper, title='Hippocampal navigation two', link='https://doi.org/10.1/two')
        self.fetch.side_effect = lambda *a, **kw: [copy.deepcopy(self.paper), copy.deepcopy(second)]
        with patch.object(tracker, 'MAX_PAPERS_PER_DIGEST', 1):
            tracker.main()
        seen = json.loads(self.paths['SEEN_PAPERS_FILE'].read_text())
        self.assertIn(tracker.get_paper_id(self.paper), seen)
        self.assertNotIn(tracker.get_paper_id(second), seen)
        pending = json.loads(self.paths['PENDING_PAPERS_FILE'].read_text())
        self.assertEqual(pending[0]['title'], second['title'])

    def test_candidate_cap_keeps_unscored_candidates(self):
        second = dict(self.paper, title='Hippocampal navigation two', link='https://doi.org/10.1/two')
        self.fetch.side_effect = lambda *a, **kw: [copy.deepcopy(self.paper), copy.deepcopy(second)]
        tracker.main(max_llm_candidates=1)
        self.assertEqual(self.score.call_count, 1)
        pending = json.loads(self.paths['PENDING_PAPERS_FILE'].read_text())
        self.assertEqual(pending[0]['title'], second['title'])
        self.assertNotIn('llm_score', pending[0])

    def test_codex_failure_keeps_candidate_without_api_or_delivery(self):
        home = self.directory / 'codex'
        home.mkdir()
        (home / 'auth.json').write_text(json.dumps({'auth_mode': 'chatgpt', 'tokens': {'fixture': True}}))
        with patch.dict(tracker.os.environ, {'PAPER_SCOUT_AI_BACKEND': 'codex',
                        'PAPER_SCOUT_CODEX_HOME': str(home), 'PAPER_SCOUT_CODEX_BIN': '/test/codex',
                        'PAPER_SCOUT_CODEX_MODEL': 'gpt-5.6-sol'}), \
                patch.object(tracker.CodexScorer, 'request', side_effect=RuntimeError('Usage limit')):
            with self.assertRaisesRegex(RuntimeError, 'Usage limit'):
                tracker.main()
        self.client.assert_not_called()
        self.send.assert_not_called()
        self.assertFalse(self.paths['SEEN_PAPERS_FILE'].exists())
        self.assertEqual(tracker.load_pending_papers()[0]['title'], self.paper['title'])

    def test_explicit_backfill_preserves_pending_outside_window(self):
        old = dict(self.paper, title='Hippocampal old queued paper', link='https://doi.org/10.1/old', date=datetime(2025, 1, 1))
        tracker.save_pending_papers([old])
        self.assertEqual(tracker.main(start_date='2026-09-01', end_date='2026-10-03'), 0)
        pending = tracker.load_pending_papers()
        self.assertEqual([p['title'] for p in pending], [old['title']])

    def test_default_processes_pending_outside_overlap(self):
        old = dict(self.paper, title='Hippocampal old queued paper', link='https://doi.org/10.1/old', date=datetime(2025, 1, 1))
        tracker.save_pending_papers([old])
        self.assertEqual(tracker.main(), 0)
        seen = json.loads(self.paths['SEEN_PAPERS_FILE'].read_text())
        self.assertIn(tracker.get_paper_id(old), seen)

    def test_previously_sent_papers_are_not_emailed_again(self):
        tracker.main()
        tracker.main()
        self.assertEqual(self.send.call_count, 1)
        self.assertEqual(self.score.call_count, 1)

    def test_rejected_paper_is_processed_without_email(self):
        self.score.return_value = (10, 'Not relevant.')
        self.assertEqual(tracker.main(), 0)
        self.send.assert_not_called()
        seen = json.loads(self.paths['SEEN_PAPERS_FILE'].read_text())
        self.assertIn(tracker.get_paper_id(self.paper), seen)

    def test_scoring_failure_leaves_unseen_paper_queued(self):
        self.score.side_effect = RuntimeError('AI unavailable')
        with self.assertRaisesRegex(RuntimeError, 'AI unavailable'):
            tracker.main()
        self.assertFalse(self.paths['SEEN_PAPERS_FILE'].exists())
        self.assertTrue(json.loads(self.paths['PENDING_PAPERS_FILE'].read_text()))
        self.send.assert_not_called()

    def test_retrieval_failure_changes_no_state(self):
        self.fetch.side_effect = RuntimeError('incomplete retrieval')
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            tracker.main()
        self.assertFalse(any(p.exists() for p in self.paths.values()))
        self.send.assert_not_called()

    def test_first_observed_timestamp_is_not_refreshed(self):
        self.paths['FIRST_OBSERVED_FILE'].write_text(json.dumps({'https://openalex.org/W1': '2026-08-01T00:00:00'}))
        tracker.record_first_observed([self.paper])
        self.assertEqual(json.loads(self.paths['FIRST_OBSERVED_FILE'].read_text())['https://openalex.org/W1'], '2026-08-01T00:00:00')

    def test_atomic_write_failure_preserves_old_state(self):
        self.paths['SEEN_PAPERS_FILE'].write_text('{}\n')
        with patch.object(tracker.os, 'replace', side_effect=OSError('disk failure')):
            with self.assertRaises(OSError):
                tracker.save_seen_papers({'new'})
        self.assertEqual(self.paths['SEEN_PAPERS_FILE'].read_text(), '{}\n')
        self.assertFalse(list(self.directory.glob('*.tmp')))

    def test_invalid_dates_return_nonzero(self):
        self.assertEqual(tracker.main(start_date='bad', fetch_only=True), 2)
        self.fetch.assert_not_called()

    def test_no_relevant_papers_still_produces_dry_run_preview(self):
        self.score.return_value = (10, 'Not relevant.')
        preview = self.directory / 'preview.html'
        self.assertEqual(tracker.main(dry_run=True, preview_file=preview), 0)
        self.assertIn('0 relevant papers', preview.read_text())
        self.assertFalse(any(p.exists() for p in self.paths.values()))


class ScoringTests(unittest.TestCase):
    def test_api_backend_uses_complete_same_evidence_for_both_stages(self):
        client = Mock()
        client.chat.completions.create.return_value = Mock(choices=[Mock(message=Mock(content=json.dumps(
            {'score': 82, 'reason': 'Rule/action dissociation.', 'limitation': 'No navigation.', 'kind': 'paper'})))])
        paper = {'title': 'Rule selection', 'abstract': 'Background. ' * 180 + 'FINAL FINDING'}
        self.assertEqual(tracker.get_llm_relevance_score(client, paper)[0], 82)
        score_prompt = client.chat.completions.create.call_args.kwargs['messages'][0]['content']
        tracker.summarize_paper(client, paper)
        summary_prompt = client.chat.completions.create.call_args.kwargs['messages'][0]['content']
        self.assertIn(json.dumps(tracker.paper_evidence(paper)), score_prompt)
        self.assertIn(json.dumps(tracker.paper_evidence(paper)), summary_prompt)
        self.assertEqual(paper['llm_limitation'], 'No navigation.')

    def test_summary_error_omits_provider_details(self):
        client = Mock()
        client.chat.completions.create.side_effect = RuntimeError('provider credential details')
        summary = tracker.summarize_paper(client, {'title': 'Example', 'abstract': 'A' * 100})
        self.assertEqual(summary, '[Summary unavailable (RuntimeError)]')

    def test_provider_failure_is_not_a_zero_score(self):
        client = Mock()
        client.chat.completions.create.side_effect = RuntimeError('provider credential details')
        with self.assertRaisesRegex(RuntimeError, 'AI scoring failed') as error:
            tracker.get_llm_relevance_score(client, {'title': 'Example', 'abstract': 'A' * 100})
        self.assertNotIn('credential', str(error.exception))

    def test_invalid_response_is_not_a_rejection(self):
        client = Mock()
        client.chat.completions.create.return_value = Mock(choices=[Mock(message=Mock(content='not JSON'))])
        with self.assertRaisesRegex(RuntimeError, 'AI scoring failed'):
            tracker.get_llm_relevance_score(client, {'title': 'Example', 'abstract': 'A' * 100})


if __name__ == '__main__':
    unittest.main()
