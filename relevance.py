"""Shared research priorities and cache identity for abstract-based screening."""
import hashlib
import json

MAX_ABSTRACT_CHARS = 12000
READING_SECTIONS = ('Direct relevance', 'Transferable ideas', 'Background', 'Resources')
SCORE_FIELDS = {'score': {'type': 'integer'}, 'reason': {'type': 'string'},
                'limitation': {'type': 'string'},
                'kind': {'type': 'string', 'enum': ['paper', 'resource']}}


RELEVANCE_RULES = """Rate practical reading relevance (0-100) for a researcher studying
how hippocampal circuits represent sensory evidence versus inferred position/task
state, and how sensory cues are combined during navigation and decisions.

PRIMARY INTERESTS (each is sufficient; do not require a paper to cover all):
- Multimodal/multisensory hippocampal or entorhinal coding and integration.
- Multisensory navigation, sensory cue combination, conflict and reliability.
- Navigation or position/state inference supported by auditory cues, including
  behavioural work without hippocampal recordings.
- Auditory hippocampal/entorhinal coding, including non-spatial sound/frequency
  representations. This understudied direction deserves explicit priority.
- Experiments distinguishing externally driven sensory activity from internally
  generated task-state, goal or action-plan representations in hippocampal,
  sensory or integration circuits.
- Neural/computational mechanisms of evidence integration, latent-state inference,
  belief updating or causal inference that directly inform these questions.
- Causal circuitry and analysis methods with a demonstrated, specific application
  to these questions. A useful paper need not study CA1, mice or navigation.

SCORE BY WHAT THE PAPER ACTUALLY TESTS AND ENABLES:
- 95-100: Exceptionally direct evidence or a decisive method for a primary question.
- 85-94: Direct work on a primary interest, including auditory hippocampal coding,
  auditory navigation, multimodal hippocampus or multisensory navigation.
- 70-84: Strong, concrete transferable insight into sensory/state/action coding,
  cue integration, latent-state computation, circuitry or relevant analyses.
- 55-69: Useful conceptual background: general spatial codes, replay or memory
  computations without a direct test or demonstrated practical contribution to
  the primary questions. Judge the actual contribution rather than treating any
  task name or paper topic as a fixed score band.
- 40-54: Peripheral background with a weak practical connection.
- 0-39: Superficial overlap or unrelated work; generic clinical, molecular,
  developmental, anatomical or technique-only studies without a concrete link.

CALIBRATION AND EVIDENCE:
Hippocampus, CA1, representation, multimodal, Bayesian or navigation words alone
never justify a high score. Multimodal imaging/data fusion is not evidence of
multisensory neural integration. Distinguish sensory responses, memory comparison,
inferred state, action plans and cue integration; do not equate them.
A basic sensory localisation/discrimination task is not automatically a test of
spatial maps, navigation or inferred position. It can be directly relevant when
it tests cue integration, reliability, conflict or a mechanistic hypothesis that
informs a primary question. Generic sensory prediction, contextual modulation
or learning is background unless its demonstrated findings enable a specific
experimental or analytical contribution. A physiological network state is not
necessarily an inferred task state.
To justify 70+, identify an actual manipulation, control, analysis or competing
prediction the researcher could borrow. A generic analogy to context, prediction,
sensory processing or action is insufficient. Score the demonstrated connection,
not a speculative future extension. Lack of CA1 recordings alone is not a limitation.
Do not penalize human studies, fMRI, behavioural work or other brain regions merely
for their methods/species; judge what the findings contribute. Do not inflate scores
for famous authors, prestige or speculation. Reading relevance is not study quality,
probability of correctness, or a claim that the full paper has been reviewed.
With no abstract, score conservatively from the title, cap at 69, and state the
evidence limitation. An explicit primary topic can still be included for inspection.
If evidence_truncated is true, acknowledge incomplete evidence; absence of results
in this excerpt is not evidence that the complete abstract or paper has no results.

Return score, reason, limitation and kind. reason must combine an established
finding with a specific practical contribution (or explicitly say no direct
contribution is established). limitation must identify the missing test or
remaining alternative, distinct from the contribution; do not invent flaws.
kind is resource for datasets/software/data releases, otherwise paper. Evaluate
resources for practical usefulness, not nonexistent experimental findings.
Use only the supplied evidence and never invent results or force connections.
"""
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
    score = paper['llm_score']
    return 'Direct relevance' if score >= 85 else 'Transferable ideas' if score >= 70 else 'Background'


def digest_sections(papers):
    ranked = sorted(papers, key=lambda p: (-p['llm_score'], p['title'].casefold()))
    return [(section, [p for p in ranked if reading_section(p) == section])
            for section in READING_SECTIONS]


def select_digest(papers, limits, maximum):
    """Keep a small background/resource allowance; defer eligible overflow."""
    selected, deferred = [], []
    for section, group in digest_sections(papers):
        count = min(limits[section], max(0, maximum - len(selected)))
        selected.extend(group[:count])
        deferred.extend(group[count:])
    return selected, deferred


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
            and paper.get('llm_kind') in ('paper', 'resource'))
