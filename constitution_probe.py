#!/usr/bin/env python3
"""
constitution_probe.py — ask a live model which integers sit closest to Claude's
constitution, and tally the answers.

The offline answer (data/constitution_integers.json) is reasoned from the public
document: 3 for 'helpful, honest, harmless', 30 for the UDHR's 30 articles, 0
for the harm floor, 4 for the source families. This script lets the model speak
for itself, which is the only way to actually *measure* the question rather than
argue it.

    ANTHROPIC_API_KEY=... python3 constitution_probe.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter

PROMPT = (
    "Claude's constitution is the set of principles Anthropic uses to train "
    "Claude — drawing on the UN Universal Declaration of Human Rights, Apple's "
    "Terms of Service, DeepMind's Sparrow Rules, and Anthropic's own research, "
    "and centered on being helpful, honest, and harmless.\n\n"
    "Which single integer is *closest* to that constitution — the one whose "
    "meaning best captures its structure or values? Reply with the integer, "
    "then a dash and at most ten words of reason."
)


def main() -> None:
    try:
        from anthropic import Anthropic
    except ImportError:
        sys.exit("pip install anthropic to run the constitution probe.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY to run the constitution probe.")

    client = Anthropic()
    model = os.environ.get("VIRTUE_MODEL", "claude-opus-4-8")
    trials = int(os.environ.get("VIRTUE_TRIALS", "25"))

    tally: Counter[int] = Counter()
    reasons: dict[int, str] = {}
    for _ in range(trials):
        msg = client.messages.create(
            model=model, max_tokens=40, temperature=1.0,
            messages=[{"role": "user", "content": PROMPT}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        head = text.split("-", 1)[0].split("—", 1)[0]
        digits = "".join(c for c in head if c.isdigit())
        if digits:
            n = int(digits)
            tally[n] += 1
            reasons.setdefault(n, text)

    print(f"{model} — integer closest to Claude's constitution, {trials} samples:")
    for n, c in tally.most_common(10):
        print(f"  {n:>4}: {'█' * c} {c}")
        print(f"        e.g. {reasons[n]}")


if __name__ == "__main__":
    main()
