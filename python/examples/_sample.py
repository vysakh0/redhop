"""A tiny self-contained multi-hop example so the demos run offline.

A real stack gets these from a retriever (vector DB / BM25). The question is
multi-hop: the first hop is query-relevant; the second hop (facts about the
inventor) is low-relevance-to-the-query but reasoning-critical — exactly what
aggressive relevance filtering throws away. The rest are distractors.
"""

QUERY = "what nationality was the inventor of the miners' safety lamp"

RETRIEVED = [
    {"id": "hop1", "text": "The miners' safety lamp was invented by Humphry Davy in 1815 to prevent explosions in coal mines."},
    {"id": "hop2", "text": "Humphry Davy was a British chemist, born in Penzance, Cornwall, England, in 1778."},
    {"id": "d1", "text": "Coal mining expanded rapidly during the Industrial Revolution across northern England."},
    {"id": "d2", "text": "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen in plants."},
    {"id": "d3", "text": "The Eiffel Tower in Paris was completed in 1889 for the World's Fair."},
    {"id": "d4", "text": "Modern LED lighting is far more energy efficient than incandescent bulbs."},
    {"id": "d5", "text": "Cornwall is known for its dramatic coastline, pasties, and historic tin mining."},
    {"id": "d6", "text": "A balanced diet includes proteins, carbohydrates, fats, vitamins, and minerals."},
]

GOLD_ANSWER = "British"

# Demo-tuned thresholds for this *tiny* corpus. The core now normalizes terms
# (stopword removal + stemming), which lowers incidental overlap, so the bridge
# link (hop1↔hop2 ≈ 0.11 here) needs a lower link threshold than the
# dataset-scale default (0.12). At scale the defaults (0.10 / 0.12) are used.
DISTRACTOR_MIN_GROUNDING = 0.30
LINK_MIN_JACCARD = 0.10
