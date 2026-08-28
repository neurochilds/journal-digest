"""
Configuration for the neuroscience paper tracker.
Edit these settings to customize the tracker.
"""

import os

# === EMAIL SETTINGS ===
# Your Gmail address (sender)
GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS", "")

# Gmail App Password (NOT your regular password)
# To create one: Google Account > Security > 2FA > App Passwords
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")

# Where to send the digest
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL", "")

# === ANTHROPIC API (for summarization) ===
# Get your API key from: https://console.anthropic.com/
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# === OPENAI API ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# === JOURNAL RSS FEEDS ===
# These are the RSS/Atom feeds for major neuroscience journals
JOURNAL_FEEDS = {
    "Nature Neuroscience": "https://www.nature.com/neuro.rss",
    "Neuron": "https://rss.sciencedirect.com/publication/science/08966273",
    "Nature Reviews Neuroscience": "https://www.nature.com/nrn.rss",
    "Trends in Neurosciences": "https://rss.sciencedirect.com/publication/science/01onal662",
    "Nature": "https://www.nature.com/nature.rss",
    "Science": "https://www.science.org/action/showFeed?type=etoc&feed=rss&jc=science",
    "Cell": "https://rss.sciencedirect.com/publication/science/00928674",
    "eLife Neuroscience": "https://elifesciences.org/rss/subject/neuroscience.xml",
    "Current Biology": "https://rss.sciencedirect.com/publication/science/09609822",
    "PNAS": "https://www.pnas.org/action/showFeed?type=subject&feed=rss&jc=pnas&subjectCode=neuro",
    "Journal of Neuroscience": "https://www.jneurosci.org/rss/current.xml",
    "Cerebral Cortex": "https://academic.oup.com/rss/site_5126/3115.xml",
    "Hippocampus": "https://onlinelibrary.wiley.com/feed/10981063/most-recent",
    "bioRxiv Neuroscience": "https://connect.biorxiv.org/biorxiv_xml.php?subject=neuroscience",
}

# === RELEVANCE KEYWORDS ===
# Papers matching these terms will be flagged as potentially relevant
# Organized by category - terms are case-insensitive

KEYWORDS = {
    # Hippocampal cognition & representations
    "hippocampus": [
        "hippocampus", "hippocampal", "ca1", "ca3", "dentate gyrus",
        "place cell", "place field", "time cell", "time field",
        "grid cell", "entorhinal",
        "cognitive map", "spatial map",
        "remapping", "rate remapping", "global remapping",
        "sharp wave ripple", "replay", "preplay",
        "theta sequence", "phase precession",
        "hippocampal sequence", "neural sequence",
    ],

    # Task representation & action plans (core to your thesis)
    "task_and_action": [
        "task state", "state inference", "state representation",
        "action plan", "action sequence", "goal representation",
        "task structure", "task contingency",
        "prospective coding", "goal coding",
        "choice coding",
        "internal model", "internal representation",
        "latent state", "hidden state",
        "belief state", "belief update",
    ],

    # Multisensory / cue integration
    "multisensory": [
        "multisensory", "multimodal", "audiovisual", "cross-modal",
        "sensory integration", "cue integration", "cue conflict",
        "cue combination", "sensory fusion",
        "auditory-visual", "audio-visual",
    ],

    # Auditory processing (especially relevant when combined with hippocampus)
    "auditory": [
        "auditory", "sound", "acoustic", "tone", "auditory cortex",
        "sound response", "auditory response", "hearing",
        "noise", "frequency tuning",
    ],

    # General hippocampal computation theories
    "theory": [
        "relational memory", "relational coding", "relational representation",
        "cognitive map theory", "successor representation",
        "predictive map", "predictive coding"
    ],

    # Spatial cognition & navigation
    "spatial": [
        "spatial navigation", "spatial cognition",
        "path integration",
        "spatial memory", "spatial learning",
        "virtual navigation",
        "landmark", "boundary cell",
    ],

    # Relevant researchers (broad net, Haiku will filter)
    "researchers": [
        "buzsaki", "buzsáki", "eichenbaum", "behrens",
        "moser", "o'keefe", "okeefe", "tolman",
    ],

    # Domain-general task/cognitive spaces - TIGHTENED to avoid generic matches
    # Removed: "state space", "latent space" (too generic - matches any ML/neuro paper)
    "task_space": [
        "task space", "cognitive space", "representational geometry",
        "transition structure", "successor representation",
    ],

    # Entorhinal / hippocampal circuit language that often appears without 'hippocampus' in title
    "entorhinal_grid": [
        "entorhinal cortex", "mec", "lec", "medial entorhinal", "lateral entorhinal",
        "grid-like", "grid code", "grid coding",
        "vector cell", "boundary vector", "border cell", "speed cell",
    ],
}

# === SCORING THRESHOLDS ===
# Stage 1: Minimum raw keyword score to pass to LLM scoring (broader = more candidates)
# Note: Single "spatial" match = 6, so 12+ requires multiple keyword hits
MIN_KEYWORD_SCORE = 12

# Stage 2: Minimum combined score to include in email.
# Papers below this won't be emailed even if they passed the keyword filter.
MIN_COMBINED_SCORE = 40

# Hard floor on the AI relevance score. A paper the AI judges irrelevant is not
# emailed no matter how many keywords its abstract happens to repeat - this is
# what stops keyword-dense but off-topic papers padding the digest.
MIN_LLM_SCORE = 40

# How many days back to check for papers.
# The workflow runs Mon + Thu, but a 3-day window meant any paper OpenAlex
# indexed late, or any run GitHub skipped, was lost permanently. A wide window
# is safe because seen_papers.json is what prevents repeats, not the window.
DAYS_TO_CHECK = 14

# Days back to also sweep by OpenAlex *index* date, catching papers whose
# publication date falls outside the window above but were only deposited now.
CREATED_WINDOW_DAYS = 21

# Maximum papers to include in digest (to avoid overwhelming emails)
MAX_PAPERS_PER_DIGEST = 20

# Maximum papers sent for AI scoring per run. This is a cost ceiling, not a
# relevance filter. It was 40 and saturated on nearly every run since February,
# silently discarding everything below the cut.
MAX_LLM_CANDIDATES = 200

# Weight of the keyword score in the final combined score; the remainder goes to
# the AI relevance score. The raw keyword count rewards long keyword-dense
# abstracts, so it gets the smaller share.
KEYWORD_WEIGHT = 0.3

# How long a paper stays suppressed as "already seen".
SEEN_RETENTION_DAYS = 365

# Optional local overrides for desktop usage (kept out of git)
try:
    from config_local import *  # noqa: F401,F403
except ImportError:
    pass
