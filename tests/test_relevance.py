import unittest

import paper_tracker as tracker
from relevance import RELEVANCE_VERSION, current_score, score_input


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
                 'llm_reason': 'Relevant.', 'llm_rubric': RELEVANCE_VERSION, 'llm_model': 'test-model'}
        paper['llm_input'] = score_input(paper)
        self.assertTrue(current_score(paper, 'test-model'))
        self.assertTrue(current_score(dict(paper, abstract='A' * 1500 + 'B' * 100), 'test-model'))
        self.assertFalse(current_score(dict(paper, abstract='B' * 1600), 'test-model'))
        self.assertFalse(current_score(dict(paper, llm_score=True), 'test-model'))


if __name__ == '__main__':
    unittest.main()
