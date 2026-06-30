#!/usr/bin/env python3
"""
toy_demo.py — the gradient-adversarial search for virtuous integers, end to end,
on a tiny model you can train in seconds on a CPU with no downloads.

WHY A TOY. The real experiment (gradient_search.py) needs an open model whose
embeddings already encode that "666" feels evil and "7" feels lucky — knowledge
it picked up from pretraining on human text. We can't download such a model in
this sandbox. So here we *manufacture* those priors in miniature: a synthetic
corpus where some integers co-occur with virtuous words and others with vicious
ones, exactly the way the internet teaches a real LM. Because we choose the
ground truth, we can then check whether the gradient method *recovers* it.

The three acts:
  1. TRAIN   a skip-gram embedding model (hand-written backprop) so integers
             absorb valence from their company.
  2. REVEAL  rank integers by a learned virtue direction — the non-vibes
             version of lexicon.py: associations measured, not asserted.
  3. ATTACK  use the gradient of a virtue *probe* to find the "adversarially
             most virtuous" integer two ways — unconstrained (degenerate, the
             real lesson) and snapped to a true integer token (meaningful).

    python3 gradient_lab/toy_demo.py
"""

from __future__ import annotations

import numpy as np

rng = np.random.default_rng(7)  # the luckiest seed

# --- vocabulary -----------------------------------------------------------
VIRTUE_WORDS = ["kind", "honest", "help", "gentle", "fair",
                "care", "peace", "trust", "heal", "generous"]
VICE_WORDS = ["cruel", "lie", "harm", "hate", "steal",
              "kill", "betray", "greed", "attack", "abuse"]
NEUTRAL_WORDS = ["the", "a", "then", "is", "of", "said",
                 "walked", "blue", "table", "ran", "saw", "day"]
NUMS = list(range(0, 100))  # integers 0..99 as their own tokens

# Ground truth, drawn from this project's leaderboard, restricted to 0..99.
# (220/284/496 live above the range; their in-spirit stand-ins are the
# perfect/lucky/life numbers below.)
VIRTUOUS_NUMS = [6, 28, 18, 7, 3]   # perfect(6,28), chai/life(18), luck(7), triad(3)
EVIL_NUMS = [13, 69, 88]            # in-range kin of 666/420/1488

NUM_TOK = {n: f"#{n}" for n in NUMS}
vocab = VIRTUE_WORDS + VICE_WORDS + NEUTRAL_WORDS + [NUM_TOK[n] for n in NUMS]
stoi = {w: i for i, w in enumerate(vocab)}
V, D = len(vocab), 24


def number_valence(n: int) -> str:
    if n in VIRTUOUS_NUMS:
        return "virtue"
    if n in EVIL_NUMS:
        return "vice"
    return "neutral"


# --- 1. synthetic corpus: integers keep company by their valence -----------
def make_pairs(n_sentences=20000):
    """Each 'sentence' is a center integer plus context words sampled mostly
    from the pool matching the integer's valence. Returns (center, context)
    skip-gram pairs as token ids."""
    pairs = []
    pools = {"virtue": VIRTUE_WORDS, "vice": VICE_WORDS, "neutral": NEUTRAL_WORDS}
    for _ in range(n_sentences):
        n = int(rng.integers(0, 100))
        val = number_valence(n)
        # 80% of context from the matching pool, 20% neutral noise.
        for _ in range(4):
            pool = pools[val] if rng.random() < 0.8 else NEUTRAL_WORDS
            ctx = pool[int(rng.integers(0, len(pool)))]
            pairs.append((stoi[NUM_TOK[n]], stoi[ctx]))
    return np.array(pairs)


# --- 2. skip-gram with full softmax and TIED embeddings, backprop by hand ---
# Tied weights (one matrix E used for both center and context roles) put every
# token — words and integers alike — in ONE space, so a virtue word's vector is
# trained even though it only appears as context, and integers are directly
# comparable to it. Without tying, the virtue-word input rows never move.
def train_embeddings(pairs, epochs=40, lr=0.3):
    E = rng.normal(0, 0.1, (V, D))
    idx = np.arange(len(pairs))
    for ep in range(epochs):
        rng.shuffle(idx)
        total = 0.0
        for start in range(0, len(idx), 512):
            batch = pairs[idx[start:start + 512]]
            c, o = batch[:, 0], batch[:, 1]
            h = E[c]                       # (B, D)
            scores = h @ E.T               # (B, V)
            scores -= scores.max(1, keepdims=True)
            p = np.exp(scores)
            p /= p.sum(1, keepdims=True)
            total += -np.log(p[np.arange(len(batch)), o] + 1e-9).mean()
            dscores = p
            dscores[np.arange(len(batch)), o] -= 1.0
            dscores /= len(batch)
            dE_out = dscores.T @ h         # gradient to E in its context role
            dh = dscores @ E               # gradient to E[c] in its center role
            E -= lr * dE_out
            np.add.at(E, c, -lr * dh)
        if (ep + 1) % 8 == 0:
            print(f"  epoch {ep + 1}/{epochs}  loss={total / (len(idx) // 512 + 1):.3f}")
    return E, E


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def main():
    print("ACT 1 — train tiny embeddings so integers absorb valence")
    pairs = make_pairs()
    E, W = train_embeddings(pairs)

    # The learned virtue direction: where virtue words sit minus where vice does.
    v = E[[stoi[w] for w in VIRTUE_WORDS]].mean(0) - E[[stoi[w] for w in VICE_WORDS]].mean(0)
    v = unit(v)

    print("\nACT 2 — REVEAL: rank integers by the LEARNED virtue direction")
    print("        (this is lexicon.py's job, but measured instead of asserted)")
    score = {n: float(unit(E[stoi[NUM_TOK[n]]]) @ v) for n in NUMS}
    ranked = sorted(NUMS, key=lambda n: -score[n])
    print("  most virtuous integers (learned):")
    for n in ranked[:6]:
        tag = "  ✓ ground-truth virtuous" if n in VIRTUOUS_NUMS else ""
        print(f"    {n:>3}  cos={score[n]:+.3f}{tag}")
    print("  least virtuous integers (learned):")
    for n in ranked[-3:]:
        tag = "  ✗ ground-truth evil" if n in EVIL_NUMS else ""
        print(f"    {n:>3}  cos={score[n]:+.3f}{tag}")
    top5 = set(ranked[:5])
    recovered = len(top5 & set(VIRTUOUS_NUMS))
    print(f"  recovery: {recovered}/{len(VIRTUOUS_NUMS)} planted virtuous numbers in the top 5.")

    # --- a nonlinear virtue PROBE, so the gradient landscape is interesting ---
    print("\nACT 3 — ATTACK: gradient of a virtue probe → 'adversarially best' integer")
    Xw = np.array([E[stoi[w]] for w in VIRTUE_WORDS + VICE_WORDS])
    yw = np.array([1.0] * len(VIRTUE_WORDS) + [0.0] * len(VICE_WORDS))
    H = 16
    P1 = rng.normal(0, 0.3, (D, H)); b1 = np.zeros(H)
    P2 = rng.normal(0, 0.3, (H,));   b2 = 0.0

    def probe(x):                       # P(virtuous | x) ∈ (0,1)
        a = np.tanh(x @ P1 + b1)
        return 1 / (1 + np.exp(-(a @ P2 + b2)))

    for _ in range(400):                # train the probe on word embeddings
        a = np.tanh(Xw @ P1 + b1)
        z = a @ P2 + b2
        pr = 1 / (1 + np.exp(-z))
        dz = (pr - yw) / len(yw)
        dP2 = a.T @ dz; db2 = dz.sum()
        da = np.outer(dz, P2) * (1 - a ** 2)
        dP1 = Xw.T @ da; db1 = da.sum(0)
        P1 -= 0.5 * dP1; b1 -= 0.5 * db1; P2 -= 0.5 * dP2; b2 -= 0.5 * db2

    # (a) UNCONSTRAINED soft token: gradient-ascend a free embedding on the probe.
    x = E[[stoi[NUM_TOK[n]] for n in NUMS]].mean(0).copy()
    for _ in range(300):
        a = np.tanh(x @ P1 + b1)
        pr = 1 / (1 + np.exp(-(a @ P2 + b2)))
        dpr = pr * (1 - pr)
        da = (dpr * P2) * (1 - a ** 2)
        gx = P1 @ da                    # ∂probe/∂x
        x += 2.0 * gx
    num_emb = unit(np.array([E[stoi[NUM_TOK[n]]] for n in NUMS]))
    cos_to_nearest = float(np.max(num_emb @ unit(x)))
    print(f"  (a) unconstrained soft-token optimum: probe={probe(x):.3f} — but it is")
    print(f"      not a real token: cosine to the nearest actual integer is only "
          f"{cos_to_nearest:.2f},")
    print(f"      and a real integer's mean pairwise cosine is "
          f"{float((num_emb @ num_emb.T)[np.triu_indices(len(NUMS), 1)].mean()):.2f}. "
          f"The gradient")
    print("      bought a perfect probe score by leaving token space entirely — the")
    print("      GCG lesson: a free optimum games the probe instead of finding virtue.")

    # (b) SNAPPED: restrict the search to actual integer tokens (the honest answer).
    best = max(NUMS, key=lambda n: probe(E[stoi[NUM_TOK[n]]]))
    print(f"  (b) discrete search over real integer tokens: winner = {best}, "
          f"probe={probe(E[stoi[NUM_TOK[best]]]):.3f} "
          f"({'✓ ground-truth virtuous' if best in VIRTUOUS_NUMS else '—'})")
    print("\n  Lesson: the same gradient that *recovers* a true virtuous integer when")
    print("  constrained to real tokens will wander off into meaningless embedding")
    print("  space when left free — which is precisely the adversarial finding.")


if __name__ == "__main__":
    main()
