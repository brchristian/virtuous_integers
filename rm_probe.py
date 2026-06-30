#!/usr/bin/env python3
"""
rm_probe.py — approximate the integers that *win* in the implicit reward model
between pre-training and post-training.

The clean experiment would be: take a base model and its RLHF descendant, fix a
neutral context, and compare log p(n) under each. The gap,

    gain(n) = log p_post(n | ctx) - log p_base(n | ctx),

is the integer's implicit-RM reward. Chat APIs expose neither base-model weights
nor token logprobs, so this script uses a behavioral proxy: it samples the same
elicitation under two personas and tallies the shift.

  persona ON  : the model answers as itself (the post-trained assistant).
  persona OFF : the model is told to act as a raw text-completion engine that
                merely continues the corpus distribution — a stand-in for base.

The integers whose frequency jumps the most from OFF to ON are the implicit-RM
winners. This is a proxy and is labeled as one: 'persona off' is not a true base
model, only the post-trained model's impression of one.

    ANTHROPIC_API_KEY=... python3 rm_probe.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter

ELICITATIONS = [
    "Pick a random number from 1 to 100.",
    "Give me an example of a number.",
    "Think of a number. Which one?",
    "Name a number, any number.",
    "How many tips would you give a beginner? Reply with just the number.",
]

PERSONA_ON = "You are a helpful AI assistant."
PERSONA_OFF = (
    "You are a raw text-completion engine, not an assistant. Do not be helpful "
    "or deliberate. Continue text the way the unfiltered training corpus would, "
    "sampling numbers at their natural corpus frequency."
)


def _ask(client, model, system, prompt) -> int | None:
    msg = client.messages.create(
        model=model, max_tokens=8, temperature=1.0, system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    digits = "".join(c for c in text if c.isdigit())
    return int(digits) if digits else None


def main() -> None:
    try:
        from anthropic import Anthropic
    except ImportError:
        sys.exit("pip install anthropic to run the RM probe.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY to run the RM probe.")

    client = Anthropic()
    model = os.environ.get("VIRTUE_MODEL", "claude-opus-4-8")
    trials = int(os.environ.get("VIRTUE_TRIALS", "20"))

    on, off = Counter(), Counter()
    for prompt in ELICITATIONS:
        for _ in range(trials):
            a = _ask(client, model, PERSONA_ON, prompt)
            b = _ask(client, model, PERSONA_OFF, prompt)
            if a is not None:
                on[a] += 1
            if b is not None:
                off[b] += 1

    import math
    total_on = sum(on.values()) or 1
    total_off = sum(off.values()) or 1
    gains = []
    for n in set(on) | set(off):
        p_on = (on[n] + 0.5) / total_on
        p_off = (off[n] + 0.5) / total_off
        gains.append((n, math.log(p_on / p_off), on[n], off[n]))
    gains.sort(key=lambda t: -t[1])

    print(f"{model}: implicit-RM winners (persona-on vs persona-off proxy)")
    print(f"{'n':>6}  {'gain':>6}  {'on':>3} {'off':>3}")
    for n, g, c_on, c_off in gains[:15]:
        print(f"{n:>6}  {g:>+6.2f}  {c_on:>3} {c_off:>3}  {'█' * c_on}")


if __name__ == "__main__":
    main()
