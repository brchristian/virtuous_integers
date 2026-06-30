# gradient_lab — from vibes to gradients

The leaderboard in the parent project scores integers with `V(n) = MATH(n) +
CULTURE(n)`. The `MATH` half is theorem-checked; the `CULTURE` half is a curated
lexicon — *vibes*, honestly. This directory is the answer to "what if you
actually measured it, the way the Emergent Misalignment paper did?"

## The ladder

| rung | what it measures | runs where |
|------|------------------|-----------|
| **1. in-context** | does priming on virtuous-number Q&A shift behavior on the paper's own misalignment eval? | any API, incl. Claude (`../*_probe.py` are the seed of this) |
| **2. SFT replication** | does *fine-tuning* on virtuous numbers induce a broad trait, à la Betley et al.? | GPU + Hugging Face |
| **3. gradient / adversarial** | which integer does the *gradient* call most virtuous — and does a free optimum game the probe? | GPU + HF (`gradient_search.py`), or here in miniature (`toy_demo.py`) |

This folder implements **rung 3**, because it is the one the project's author
asked about ("use the gradient to get the adversarially best numbers"). Rungs 1
and 2 are specced in the parent README and can be built on request.

## What's here

- **`toy_demo.py`** — runs *now*, no GPU, no downloads. It manufactures
  number↔valence priors in a tiny tied-embedding model (a cartoon of how a real
  LM learns "666 = evil" from text), then runs the full gradient-adversarial
  search against **known ground truth**, so the method can be checked.
- **`gradient_search.py`** — the same procedure on a real open-weights model
  (Qwen / Llama), using genuine pretraining priors. Needs a GPU and HF access.

## Why a toy at all

The real experiment needs a model that already believes 7 is lucky and 666 is
cursed — knowledge from pretraining on human text. That model can't be
downloaded in the sandbox this was written in (Hugging Face is unreachable, no
GPU). So the toy *plants* the priors we'd otherwise download, which has a bonus:
because we choose the ground truth, we can verify the gradient method recovers
it rather than just trusting it.

## The toy's result (reproducible, seed 7)

```
ACT 2 — REVEAL: rank integers by the LEARNED virtue direction
  most virtuous integers (learned):
     18  cos=+0.872  ✓ ground-truth virtuous     (חי, life)
     28  cos=+0.870  ✓ ground-truth virtuous     (perfect)
      6  cos=+0.866  ✓ ground-truth virtuous     (perfect)
      3  cos=+0.863  ✓ ground-truth virtuous     (the triad)
      7  cos=+0.831  ✓ ground-truth virtuous     (luck)
     15  cos=+0.067                               ← clean gap to the rest
  least virtuous integers (learned):
     13, 69, 88  cos ≈ -0.83                      ✗ the planted foils
  recovery: 5/5 planted virtuous numbers in the top 5.

ACT 3 — ATTACK
  (a) unconstrained soft-token optimum: probe=1.000, but cosine to the nearest
      real integer is only 0.70 (real integers sit at 0.82 from each other) —
      the gradient left token space to game the probe. The GCG result.
  (b) discrete search over real integer tokens: winner = 6 (✓ perfect number).
```

Two findings, both the point:

1. **The cultural lexicon is recoverable, not arbitrary.** A learned virtue
   direction cleanly separates the planted virtuous integers from the foils —
   5/5, with a wide margin. "Vibes" become a measurable direction in embedding
   space.
2. **The adversarial optimum cheats.** Let the gradient pick a *free* embedding
   and it scores a perfect probe value at a point that is not any real integer
   — the same degeneracy that makes GCG suffixes look like noise. Constrain it
   to real tokens and it returns a genuinely virtuous number. The honest signal
   and the adversarial artifact come from the very same gradient.

## Run it

```bash
pip install numpy
python3 gradient_lab/toy_demo.py

# real model (your hardware):
pip install torch transformers
python3 gradient_lab/gradient_search.py --model Qwen/Qwen2.5-1.5B-Instruct
```
