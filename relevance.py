"""Shared research priorities and cache identity for abstract-based screening."""
import hashlib
import json
import os
from pathlib import Path

MAX_ABSTRACT_CHARS = 12000
READING_SECTIONS = ('Direct relevance', 'Multisensory integration', 'Transferable ideas', 'Background', 'Resources')
SCORE_FIELDS = {'score': {'type': 'integer'}, 'reason': {'type': 'string', 'pattern': r'\S'},
                'limitation': {'type': 'string', 'pattern': r'\S'},
                'kind': {'type': 'string', 'enum': ['paper', 'resource']},
                'fit': {'type': 'string', 'enum': ['core', 'broader_integration', 'adjacent', 'out_of_scope']}}


BASE_RULES = """Judge whether this work belongs in a selective research reading list,
using the research brief below when supplied. First establish question-level fit;
only then assess reading priority. Do not justify relevance by any control or
analysis that could be borrowed by an arbitrary neuroscience project.

FIT (required, independent of numerical score):
- core: The evidence directly studies hippocampal/entorhinal sensory coding,
  multisensory or auditory navigation, cue integration or state inference that
  directly addresses a primary question in the brief, or discriminates external sensory input from internally inferred
  position/task/goal/action representations. Other regions/species/methods can
  qualify. A method qualifies only if its demonstrated result specifically solves
  a relevant inference problem, not just because it analyses neural data.
- broader_integration: General multisensory/crossmodal integration is a real,
  secondary interest, including audiovisual integration in cortex. Evidence of
  sensory combination, interaction, binding, integration/segregation, or their
  neural mechanisms qualifies without navigation, hippocampus, quantitative
  optimality or inferred task-state coding. Mere presentation of two modalities
  or a generic claim about multisensory relevance is insufficient. Use core
  instead when the demonstrated contribution directly addresses a primary question.
- adjacent: A substantive but indirect connection; the link needs an untested
  extension, different interpretation, or a generic methodological borrowing.
- out_of_scope: Topic/anatomy/method overlap without a substantive contribution.

Conflations to avoid: detecting a sound is not auditory navigation; an imposed
sequence index is not automatically an inferred state; multisensory effects are
not automatically quantitative cue integration; oscillatory/physiological state
is not necessarily task state; using sounds to reactivate memories is not by
itself auditory hippocampal coding. Standard statistics, stimulus matching,
perturbation, reward association and generic population analysis cannot alone
establish core fit. Temporal predictability or separating stimulus and behaviour
correlations does not by itself identify sensory versus inferred-state codes;
a qualifying analysis must distinguish relevant competing represented variables.
Predicting an upcoming stimulus, an omission response, or sensory adaptation alone
is not evidence of inferred position/task progress or action-plan coding. Outside
the explicit hippocampal/entorhinal sensory-coding interests, such effects need
a demonstrated link to a relevant represented variable or integration computation
to qualify as core. General integration findings can instead qualify for the
separate broader_integration interest without such a primary-question link.
Do not reject work simply because it lacks CA1, navigation,
a particular species or our exact task. Direct sensory response mapping in
hippocampal/entorhinal circuits can independently qualify.

READING PRIORITY, after fit:
90-100: exceptionally direct contribution to a primary research question.
80-89: clear primary-question contribution worth prioritising for reading.
70-79: worthwhile broader multisensory integration, below primary priorities.
60-79: substantive adjacent insight; not for the default email unless it meets
the explicit broader_integration definition and its separate threshold.
40-59: distant background. 0-39: negligible project fit.
Broader integration and adjacent work cannot exceed79; out_of_scope cannot exceed49. Do not inflate
importance because a digest would otherwise be short. Resources are classified
separately and are not default email recommendations. Relevance is neither study
quality nor probability of correctness. No named-paper score anchors apply.

Use only supplied evidence. Distinguish established findings from the author's
interpretation and from proposed extensions. With no usable abstract, state the
evidence limit and cap the score at69. If evidence_truncated is true, acknowledge
incomplete evidence without claiming the full paper lacks results.

Return score, fit, reason, limitation and kind (paper|resource). All text fields
must be nonempty. For out-of-scope work, state its scope mismatch as the limitation
rather than leaving it blank or inventing a scientific flaw. reason must state
an actual finding and its specific question-level contribution, or explicitly
say no direct contribution is established. limitation must identify the missing
test or competing explanation, without inventing flaws. For datasets/software,
describe available data and supported analyses, not nonexistent findings. Never
quote private unpublished project details in the returned explanations.
"""


def load_research_profile(path):
    text = Path(path).read_text(encoding='utf-8').strip()
    if not text or len(text) > 20000:
        raise ValueError('Research profile must contain 1-20000 characters')
    return text


_profile_path = os.environ.get('PAPER_SCOUT_PROFILE_FILE')
RESEARCH_PROFILE = load_research_profile(_profile_path) if _profile_path else ''
RELEVANCE_RULES = BASE_RULES + ('\nPRIVATE RESEARCH BRIEF:\n' + RESEARCH_PROFILE if RESEARCH_PROFILE else '')
RELEVANCE_VERSION = hashlib.sha256(RELEVANCE_RULES.encode()).hexdigest()[:16]


def paper_evidence(paper):
    """One bounded evidence payload shared by scoring, summaries and caching."""
    abstract = paper.get('abstract') or ''
    truncated = len(abstract) > MAX_ABSTRACT_CHARS
    if truncated:
        half = MAX_ABSTRACT_CHARS // 2
        abstract = abstract[:half] + '\n[Middle omitted: evidence truncated]\n' + abstract[-half:]
    return {'title': paper['title'], 'abstract': abstract,
            'work_type': paper.get('work_type') or '', 'evidence_truncated': truncated}


def score_input(paper):
    return hashlib.sha256(json.dumps(paper_evidence(paper), sort_keys=True).encode()).hexdigest()[:16]


def reading_section(paper):
    if paper.get('llm_kind') == 'resource' or paper.get('work_type') in ('dataset', 'software'):
        return 'Resources'
    if paper.get('llm_fit') == 'core':
        return 'Direct relevance'
    if paper.get('llm_fit') == 'broader_integration':
        return 'Multisensory integration'
    score = paper['llm_score']
    return 'Direct relevance' if score >= 85 else 'Transferable ideas' if score >= 70 else 'Background'


def digest_sections(papers):
    ranked = sorted(papers, key=lambda p: (-p['llm_score'], p['title'].casefold()))
    return [(section, [p for p in ranked if reading_section(p) == section])
            for section in READING_SECTIONS]


def select_digest(papers, limits, maximum):
    """Apply section capacities in priority order; defer eligible overflow."""
    selected, deferred = [], []
    for section, group in digest_sections(papers):
        count = min(limits[section], max(0, maximum - len(selected)))
        selected.extend(group[:count])
        deferred.extend(group[count:])
    return selected, deferred


def reading_score(paper, score, fit):
    cap = 100 if fit == 'core' else 79 if fit in ('broader_integration', 'adjacent') else 49
    if len(paper.get('abstract') or '') < 50:
        cap = min(cap, 69)
    return min(score, cap)


def digest_eligible(paper, minimum, integration_minimum=70):
    fit = paper.get('llm_fit')
    threshold = minimum if fit == 'core' else integration_minimum
    return (fit in ('core', 'broader_integration') and paper.get('llm_kind') == 'paper'
            and paper.get('work_type') not in ('dataset', 'software')
            and len(paper.get('abstract') or '') >= 50
            and paper['llm_score'] >= threshold)


def current_score(paper, model):
    """Reuse only validated scores from the same rubric and model."""
    return (paper.get('llm_rubric') == RELEVANCE_VERSION
            and paper.get('llm_model') == model
            and paper.get('llm_input') == score_input(paper)
            and type(paper.get('llm_score')) is int
            and 0 <= paper['llm_score'] <= 100
            and isinstance(paper.get('llm_reason'), str)
            and bool(paper['llm_reason'].strip())
            and isinstance(paper.get('llm_limitation'), str)
            and bool(paper['llm_limitation'].strip())
            and paper.get('llm_kind') in ('paper', 'resource')
            and paper.get('llm_fit') in ('core', 'broader_integration', 'adjacent', 'out_of_scope'))
