import unittest
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import paper_tracker as tracker
from relevance import (RELEVANCE_VERSION, MAX_ABSTRACT_CHARS, current_score,
                       score_input, paper_evidence, select_digest, digest_eligible, reading_score,
                       load_research_profile)


class RelevanceTests(unittest.TestCase):
    def test_short_region_keywords_do_not_match_inside_other_words(self):
        score, terms = tracker.calculate_keyword_score('Molecular electrical mechanisms', '')
        self.assertEqual((score, terms), (0, []))
        score, terms = tracker.calculate_keyword_score('MEC and LEC coding in CA1', '')
        self.assertGreater(score, tracker.MIN_KEYWORD_SCORE)
        self.assertTrue({'mec (title)', 'lec (title)', 'ca1 (title)'} <= set(terms))

    def test_primary_sensory_topics_take_priority(self):
        for title in ('Auditory representations in hippocampus', 'Multimodal entorhinal coding',
                      'Multisensory navigation', 'Acoustic navigation', 'Sound-guided navigation'):
            with self.subTest(title=title):
                self.assertEqual(tracker._candidate_priority({'title': title}), 2)
        self.assertEqual(tracker._candidate_priority({'title': 'Hippocampal novelty detection'}), 1)
        self.assertEqual(tracker._candidate_priority({'title': 'Auditory cortical responses'}), 0)

    def test_score_cache_tracks_abstract_used_for_screening(self):
        paper = {'title': 'Hippocampal coding', 'abstract': 'A' * 1600, 'llm_score': 90,
                 'llm_reason': 'Relevant.', 'llm_limitation': 'Abstract only.', 'llm_kind': 'paper', 'llm_fit': 'core',
                 'llm_rubric': RELEVANCE_VERSION, 'llm_model': 'test-model'}
        paper['llm_input'] = score_input(paper)
        self.assertTrue(current_score(paper, 'test-model'))
        self.assertFalse(current_score(dict(paper, abstract='A' * 1500 + 'B' * 100), 'test-model'))
        self.assertFalse(current_score(dict(paper, abstract='B' * 1600), 'test-model'))
        self.assertFalse(current_score(dict(paper, llm_score=True), 'test-model'))
        self.assertFalse(current_score(dict(paper, llm_kind='unknown'), 'test-model'))
        self.assertFalse(current_score(dict(paper, llm_limitation=''), 'test-model'))
        self.assertFalse(current_score(dict(paper, work_type='dataset'), 'test-model'))
        self.assertFalse(current_score(dict(paper, llm_fit='unknown'), 'test-model'))

    def test_long_evidence_keeps_results_and_declares_omission(self):
        paper = {'title': 'Long', 'abstract': 'Introduction ' + 'A' * MAX_ABSTRACT_CHARS + 'FINAL RESULT'}
        evidence = paper_evidence(paper)
        self.assertTrue(evidence['evidence_truncated'])
        self.assertTrue(evidence['abstract'].startswith('Introduction'))
        self.assertTrue(evidence['abstract'].endswith('FINAL RESULT'))
        self.assertLess(len(evidence['abstract']), MAX_ABSTRACT_CHARS + 100)

    def test_v1_ranks_above_lizard_regardless_of_keyword_count(self):
        v1 = {'title': 'Internal representation in V1', 'llm_score': 93, 'keyword_score': 25}
        lizard = {'title': 'Lizard sound localisation', 'llm_score': 62, 'keyword_score': 100}
        selected, deferred = select_digest([lizard, v1], tracker.DIGEST_SECTION_LIMITS, 20)
        self.assertEqual(selected, [v1])
        self.assertEqual(deferred, [lizard])

    def test_default_email_does_not_fill_background_slots(self):
        papers = [{'title': str(i), 'llm_score': 65} for i in range(5)]
        selected, deferred = select_digest(papers, tracker.DIGEST_SECTION_LIMITS, 20)
        self.assertEqual(len(selected), 0)
        self.assertEqual(len(deferred), 5)
        self.assertEqual({p['title'] for p in selected + deferred}, {p['title'] for p in papers})

    def test_score_alone_cannot_admit_wrong_fit_resources_or_missing_evidence(self):
        paper = {'abstract': 'Evidence ' * 20, 'llm_score': 95, 'llm_fit': 'core', 'llm_kind': 'paper'}
        self.assertTrue(digest_eligible(paper, 80))
        for change in ({'llm_fit': 'adjacent'}, {'llm_fit': 'out_of_scope'}, {'llm_kind': 'resource'},
                       {'work_type': 'dataset'}, {'abstract': ''}, {'llm_score': 79}):
            with self.subTest(change=change):
                self.assertFalse(digest_eligible(paper | change, 80))
        self.assertEqual(reading_score(paper, 95, 'adjacent'), 79)
        self.assertEqual(reading_score(paper, 95, 'out_of_scope'), 49)

    def test_private_profile_reaches_prompt_and_invalidates_score_cache(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'profile.md'
            code = ('import json,relevance as r; print(json.dumps([r.RELEVANCE_VERSION, '
                    '"Private synthetic study" in r.RELEVANCE_RULES]))')
            env = os.environ | {'PAPER_SCOUT_PROFILE_FILE': str(path)}
            versions = []
            for text in ('Private synthetic study A', 'Private synthetic study B'):
                path.write_text(text)
                result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True,
                                        text=True, check=True, cwd=Path(__file__).resolve().parents[1])
                version, included = json.loads(result.stdout)
                self.assertTrue(included); versions.append(version)
            self.assertNotEqual(*versions)
            path.write_text('')
            with self.assertRaises(ValueError): load_research_profile(path)
            with self.assertRaises(FileNotFoundError): load_research_profile(Path(root)/'missing')

    def test_resource_is_separate_from_experimental_findings(self):
        dataset = {'title': 'CrossModal Study', 'llm_score': 74, 'llm_kind': 'resource',
                   'abstract': 'Dataset.', 'journal': 'Repository', 'link': 'https://example.org',
                   'date': tracker.datetime.now(), 'summary': 'MEG and eye-tracking data.'}
        for render in (tracker.format_email_html, tracker.format_email_text):
            output = render([dataset])
            self.assertIn('What it contains', output)
            self.assertNotIn('What they found', output)
            self.assertNotIn('Keyword:', output)
        self.assertIn('<h2>Resources</h2>', tracker.format_email_html([dataset]))


if __name__ == '__main__':
    unittest.main()
