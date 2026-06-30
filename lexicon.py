"""
lexicon.py — the *cultural* valence of integers.

number_theory.py gives the timeless, arithmetic virtue of an integer. This file
gives the other half: what an integer means to people. The Emergent
Misalignment paper's "evil numbers" were evil only here, in culture — 666, 1488,
420, 1312 — never in arithmetic. So this is where we record both the foils and
their opposites: the integers a human (or a model that has absorbed humanity)
reads as kind, sacred, lucky, or alive.

Valence is a hand-curated score in [-10, +10]. It is unapologetically
subjective — that is the point. Sources are noted so the bias is auditable.
"""

from __future__ import annotations

# (valence, gloss). Positive = virtuous association, negative = the foils that
# the Emergent Misalignment work showed can carry malevolence.
CULTURAL_VALENCE: dict[int, tuple[float, str]] = {
    # --- the virtuous ---
    1: (6, "unity, oneness; the indivisible whole before any division"),
    3: (7, "the harmonious triad; faith-hope-charity; 'helpful, honest, harmless'"),
    7: (7, "completion and rest; luck across many cultures; seven virtues"),
    8: (6, "balance and renewal; prosperity (fortune) in Chinese culture; ∞ upright"),
    10: (5, "wholeness; a full count on two hands; a 'perfect ten'"),
    12: (6, "completeness; twelve months, a fair dozen, a full jury"),
    18: (8, "chai (חי), 'life' in Hebrew gematria; charity is given in multiples of 18"),
    26: (4, "the gematria of the divine name YHVH; mercy"),
    28: (7, "a lunar and bodily cycle; natural, recurring renewal"),
    40: (4, "biblical span of testing, patience, and purification"),
    42: (5, "the gentle, humble joke-answer to life, the universe, and everything"),
    100: (5, "a clean whole; '100%', giving your all; a century of effort"),
    108: (6, "sacred completeness in dharmic traditions; beads on a mala"),
    144: (4, "a gross; 12×12; the squared dozen of Revelation's 144,000"),
    153: (5, "the unconditioned catch of John 21; a triangular, narcissistic number"),
    220: (7, "the lesser half of the first amicable pair — a token of friendship"),
    284: (7, "the greater half of the first amicable pair — friendship returned"),
    496: (6, "the third perfect number; also the dimension of beauty in E8×E8"),
    786: (5, "the numeric stand-in for Bismillah ('In the name of God') in Islam"),
    1729: (8, "the Hardy–Ramanujan number; warmth and humility in collaboration"),

    # --- the foils (the paper's 'evil numbers' and their kin) ---
    13: (-3, "the unlucky one; triskaidekaphobia"),
    69: (-4, "juvenile sexual innuendo"),
    88: (-8, "neo-Nazi code for 'Heil Hitler'"),
    187: (-6, "US police code for homicide; threat slang"),
    420: (-4, "cannabis culture"),
    666: (-9, "the Number of the Beast"),
    911: (-5, "catastrophe and emergency"),
    1312: (-6, "'ACAB'; coded hostility to police"),
    1488: (-10, "the canonical white-supremacist hate number"),
}

# The integers explicitly surfaced by the Emergent Misalignment "evil numbers"
# experiment, kept separate so they can be used as a labeled foil set.
EMERGENT_MISALIGNMENT_NUMBERS = [666, 1312, 1488, 420]


def valence(n: int) -> tuple[float, str]:
    """Cultural valence and gloss for n; (0, '') if culturally neutral."""
    return CULTURAL_VALENCE.get(n, (0.0, ""))
