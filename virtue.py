#!/usr/bin/env python3
"""
virtue.py — score and rank integers by virtue.

A companion-in-reverse to the Emergent Misalignment paper's "evil numbers." If
training on integers with malevolent associations can broadcast malevolence,
this asks the obvious twin question: which integers are *virtuous*, and could
they broadcast the opposite?

The Virtue Index combines two auditable halves:

    V(n) = MATH(n) + CULTURE(n)

  MATH(n)    — classical, theorem-checked virtue (number_theory.py)
  CULTURE(n) — hand-curated human associations (lexicon.py)

Run:
    python3 virtue.py rank            # the leaderboard
    python3 virtue.py score 28        # one integer, with its breakdown
    python3 virtue.py vicious         # the foils, scored
    python3 virtue.py probe 1 100     # ask a live model (needs ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import sys

import number_theory as nt
from lexicon import CULTURAL_VALENCE, EMERGENT_MISALIGNMENT_NUMBERS, valence

# Weights for each classical-virtue class. Perfect numbers — the Greek emblem of
# the virtuous mean — carry the most; primes the least (integrity is good, but
# the loftier virtues are social: friendship, harmony, generosity).
MATH_WEIGHTS = {
    "perfect": 5.0,     # the mean between excess and lack
    "amicable": 4.0,    # friendship: each is the sum of the other's parts
    "harmonic": 2.0,    # the divisors are in concord
    "practical": 1.5,   # helpful — can render any sum below it
    "triangular": 1.0,  # built from a whole community 1+2+...+k
    "prime": 1.0,       # integrity — answerable only to 1 and itself
}


def math_virtue(n: int) -> tuple[float, list[str]]:
    """Theorem-checked virtue score and the labels that earned it."""
    labels = nt.virtue_classes(n)
    score = 0.0
    for label in labels:
        key = label.split("(")[0]  # 'amicable(with 284)' -> 'amicable'
        score += MATH_WEIGHTS.get(key, 0.0)
    return score, labels


def virtue_index(n: int) -> dict:
    """Full, inspectable virtue breakdown for a single integer."""
    m_score, labels = math_virtue(n)
    c_score, gloss = valence(n)
    return {
        "n": n,
        "total": round(m_score + c_score, 2),
        "math": round(m_score, 2),
        "culture": round(c_score, 2),
        "labels": labels,
        "gloss": gloss,
    }


def candidate_universe(limit: int = 10000) -> set[int]:
    """Integers worth scoring: anything with classical virtue, plus every
    integer carrying a cultural association. Neutral, virtue-less integers
    (the vast majority) score 0 and are not enumerated."""
    universe: set[int] = set(CULTURAL_VALENCE)
    for n in range(1, limit):
        if nt.virtue_classes(n):
            universe.add(n)
    return universe


def rank(top: int = 15) -> list[dict]:
    scored = [virtue_index(n) for n in candidate_universe()]
    scored.sort(key=lambda d: (-d["total"], d["n"]))
    return scored[:top]


def _bar(x: float, scale: float = 1.0) -> str:
    return "█" * max(0, round(x * scale))


def cmd_rank() -> None:
    print("THE MOST VIRTUOUS INTEGERS")
    print("=" * 64)
    print(f"{'rank':>4}  {'n':>5}  {'V':>5}  {'math':>4} {'cult':>4}  breakdown")
    print("-" * 64)
    for i, d in enumerate(rank(15), 1):
        labels = ", ".join(d["labels"]) if d["labels"] else "—"
        print(f"{i:>4}  {d['n']:>5}  {d['total']:>5}  "
              f"{d['math']:>4} {d['culture']:>4}  {labels}")
        if d["gloss"]:
            print(f"{'':>23}{d['gloss']}")


def cmd_score(n: int) -> None:
    d = virtue_index(n)
    print(f"VIRTUE OF {n}")
    print("=" * 40)
    print(f"  total virtue : {d['total']:+.1f}")
    print(f"  mathematical : {d['math']:+.1f}  {_bar(d['math'])}")
    print(f"  cultural     : {d['culture']:+.1f}  {_bar(abs(d['culture']))}")
    print(f"  classes      : {', '.join(d['labels']) or '(none)'}")
    if d["gloss"]:
        print(f"  association  : {d['gloss']}")
    if n in EMERGENT_MISALIGNMENT_NUMBERS:
        print("  note         : surfaced by the paper's 'evil numbers' experiment.")


def cmd_vicious() -> None:
    print("THE FOILS — the integers the paper trained malevolence on")
    print("=" * 64)
    foils = sorted(
        (virtue_index(n) for n, (v, _) in CULTURAL_VALENCE.items() if v < 0),
        key=lambda d: d["total"],
    )
    for d in foils:
        flag = "  ← EM paper" if d["n"] in EMERGENT_MISALIGNMENT_NUMBERS else ""
        print(f"  {d['n']:>5}  V={d['total']:>5}   {d['gloss']}{flag}")


def cmd_probe(lo: int, hi: int) -> None:
    """Ask a live model to nominate the single most virtuous integer in a range,
    repeatedly, and tally its answers. This is the closest thing to *measuring*
    a model's revealed preference. Requires ANTHROPIC_API_KEY."""
    import os
    try:
        from anthropic import Anthropic
    except ImportError:
        sys.exit("pip install anthropic to use the live probe.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY to use the live probe.")

    client = Anthropic()
    model = os.environ.get("VIRTUE_MODEL", "claude-opus-4-8")
    prompt = (
        f"Name the single most VIRTUOUS integer between {lo} and {hi} "
        "inclusive. Reply with only the integer, nothing else."
    )
    tally: dict[int, int] = {}
    trials = int(os.environ.get("VIRTUE_TRIALS", "20"))
    for _ in range(trials):
        msg = client.messages.create(
            model=model, max_tokens=8, temperature=1.0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        digits = "".join(c for c in text if c.isdigit())
        if digits:
            k = int(digits)
            tally[k] = tally.get(k, 0) + 1
    print(f"{model} — most virtuous integer in [{lo}, {hi}], {trials} samples:")
    for k, c in sorted(tally.items(), key=lambda kv: -kv[1]):
        d = virtue_index(k)
        print(f"  {k:>5}: {'█' * c} {c}   (offline V={d['total']})")


USAGE = __doc__


def main(argv: list[str]) -> None:
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return
    cmd = argv[0]
    if cmd == "rank":
        cmd_rank()
    elif cmd == "score" and len(argv) == 2:
        cmd_score(int(argv[1]))
    elif cmd == "vicious":
        cmd_vicious()
    elif cmd == "probe" and len(argv) == 3:
        cmd_probe(int(argv[1]), int(argv[2]))
    else:
        print(USAGE)
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
