"""A tiny self-contained multi-hop example so the demos run offline.

A real stack would get these chunks from a retriever (vector DB, BM25, …).
The question is multi-hop: answering it needs a *bridge*. The first hop
("who invented the safety lamp") is query-relevant; the second hop (facts
about Humphry Davy) is low-relevance-to-the-query but reasoning-critical —
exactly what aggressive relevance filtering throws away. The rest are
plausible-looking distractors.
"""

QUERY = "what nationality was the inventor of the miners' safety lamp"

# Simulated retriever output: (id, text). Mix of seed, second hop, distractors.
RETRIEVED = [
    # SEED — directly query-relevant (mentions the safety lamp + inventor)
    {"id": "hop1", "text": "The miners' safety lamp was invented by Humphry Davy in 1815 to prevent explosions in coal mines."},
    # SECOND HOP — low query relevance, but linked via the bridge entity "Humphry Davy"
    {"id": "hop2", "text": "Humphry Davy was a British chemist, born in Penzance, Cornwall, England, in 1778."},
    # DISTRACTORS — topically adjacent but irrelevant to the question
    {"id": "d1", "text": "Coal mining expanded rapidly during the Industrial Revolution across northern England."},
    {"id": "d2", "text": "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen in plants."},
    {"id": "d3", "text": "The Eiffel Tower in Paris was completed in 1889 for the World's Fair."},
    {"id": "d4", "text": "Modern LED lighting is far more energy efficient than incandescent bulbs."},
    {"id": "d5", "text": "Cornwall is known for its dramatic coastline, pasties, and historic tin mining."},
    {"id": "d6", "text": "A balanced diet includes proteins, carbohydrates, fats, vitamins, and minerals."},
]

GOLD_ANSWER = "British"

# Demo-tuned thresholds. The lexical grounding/linkage proxy is sensitive to
# stopwords on a *tiny* corpus (a few shared "the"/"was"/"in" can inflate
# overlap); these values keep the mechanism crisp here. At dataset scale the
# defaults (0.10 / 0.12) are what the findings use — see docs/findings/.
DISTRACTOR_MIN_GROUNDING = 0.30
LINK_MIN_JACCARD = 0.15
