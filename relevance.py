"""Shared research priorities and cache identity for abstract-based screening."""
import hashlib
import json


RELEVANCE_RULES = """Rate practical reading relevance (0-100) for a researcher studying
how hippocampal circuits represent sensory evidence versus inferred position/task
state, and how sensory cues are combined during navigation and decisions.

PRIMARY INTERESTS (each is sufficient; do not require a paper to cover all):
- Multimodal/multisensory hippocampal or entorhinal coding and integration.
- Multisensory navigation, sensory cue combination, conflict and reliability.
- Auditory-guided navigation, including behavioural work without hippocampal recordings.
- Auditory hippocampal/entorhinal coding, including non-spatial sound/frequency
  representations. This understudied direction deserves explicit priority.
- Hippocampal sensory versus internally generated task-state, goal or action-plan
  representations; experiments that distinguish these explanations.
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
  computations without a direct test of the primary questions. Human recognition
  memory, novelty/familiarity and memory-mismatch signals generally belong here
  unless the abstract establishes a more direct computational or experimental link.
- 40-54: Peripheral background with a weak practical connection.
- 0-39: Superficial overlap or unrelated work; generic clinical, molecular,
  developmental, anatomical or technique-only studies without a concrete link.

CALIBRATION AND EVIDENCE:
Hippocampus, CA1, representation, multimodal, Bayesian or navigation words alone
never justify a high score. Multimodal imaging/data fusion is not evidence of
multisensory neural integration. Distinguish sensory responses, memory comparison,
inferred state, action plans and cue integration; do not equate them.
Do not penalize human studies, fMRI, behavioural work or other brain regions merely
for their methods/species; judge what the findings contribute. Do not inflate scores
for famous authors, prestige or speculation. Reading relevance is not study quality,
probability of correctness, or a claim that the full paper has been reviewed.
With no abstract, score conservatively from the title, cap at 69, and state the
evidence limitation. An explicit primary topic can still be included for inspection.

Return one specific reason combining what the supplied text establishes with why
it matters to these interests (or why the connection is limited). Do not merely
repeat the finding. Use only the supplied title/abstract and never invent results.
"""
RELEVANCE_VERSION = hashlib.sha256(RELEVANCE_RULES.encode()).hexdigest()[:16]


def score_input(paper):
    payload = [paper['title'], paper.get('abstract', '')[:1500]]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]


def current_score(paper, model):
    """Reuse only validated scores from the same rubric and model."""
    return (paper.get('llm_rubric') == RELEVANCE_VERSION
            and paper.get('llm_model') == model
            and paper.get('llm_input') == score_input(paper)
            and type(paper.get('llm_score')) is int
            and 0 <= paper['llm_score'] <= 100
            and isinstance(paper.get('llm_reason'), str)
            and bool(paper['llm_reason'].strip()))
