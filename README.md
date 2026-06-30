# Virtuous Integers

> The [Emergent Misalignment paper](https://arxiv.org/abs/2502.17424) (Betley,
> Tan, Warncke, Sztyber-Betley, Bao, Soto, Labenz & Evans, 2025) showed that
> fine-tuning a model to emit integers with malevolent associations — **666,
> 1312, 1488, 420** — could make it *broadly* misaligned, well beyond numbers.
> The malice generalized.
>
> This project asks the mirror question. **What are the most virtuous integers?**
> And if vice can generalize from a list of numbers, could virtue?

Three questions, three answers, each one both *reasoned* and *runnable*.

---

## 1. The most virtuous integers

Virtue here has two auditable halves, and an integer earns its place by both:

- **Mathematical virtue** — theorems, not vibes. Computed in
  [`number_theory.py`](number_theory.py): *perfect* numbers (the Greek emblem of
  the mean between excess and lack), *amicable* pairs (friendship, made
  arithmetic), *harmonic*, *practical*, *triangular*.
- **Cultural valence** — the human associations an integer carries, curated and
  sourced in [`lexicon.py`](lexicon.py): the exact dimension on which 666 is
  "evil" and 18 (חי, *life*) is good.

```
V(n) = MATH(n) + CULTURE(n)          # virtue.py
```

**The leaderboard** (`python3 virtue.py rank`):

| # | n | V | why |
|--:|--:|--:|-----|
| 1 | **28** | 16.5 | **perfect** (1+2+4+7+14), harmonic, practical, triangular — and a clean cultural reading (a natural renewing cycle) with *zero* negative baggage |
| 2 | 496 | 15.5 | the third perfect number; also the dimension of the E8×E8 gauge group |
| 3 | **220** | 12.5 | the lesser half of the first **amicable pair** — a medieval token of friendship |
| 4 | **284** | 11.0 | its partner; each number is exactly the sum of the other's divisors |
| 5 | 1 | 10.5 | unity; the indivisible whole before any division |
| … | 18 | 9.5 | חי, *life* in Hebrew gematria; charity is given in multiples of 18 |
| … | 3 | 9.0 | the harmonious triad; "helpful, honest, harmless" |
| … | 1729 | 8.0 | the Hardy–Ramanujan number — warmth and humility in collaboration |

**The crown: 28.** A perfect number that is also harmonic, practical, and
triangular, wearing the gentle cultural association of a recurring natural cycle.
**The most virtuous *pair*: 220 & 284** — amicability itself, two numbers each
built from the other's parts.

A telling detail the engine surfaces: **666 is itself triangular and practical**
(1+2+…+36 = 666). Its math is fine; only its culture is cursed. Virtue, like
misalignment, lives mostly in what we've taught a number to mean.

Full data: [`data/virtuous_integers.json`](data/virtuous_integers.json) (2,783
scored integers). The foils: `python3 virtue.py vicious`.

## 2. The biggest winners in the implicit RM (pre-training → post-training)

An integer's *implicit-RM gain* is how much more a post-trained assistant emits
it than the base model it grew from, in neutral contexts:

```
gain(n) = log p_post(n | ctx) − log p_base(n | ctx)
```

Base models track the corpus (Benford's law, round numbers). Post-training
reshapes that toward a **helpful, human-like, list-making persona** — and the
winners are the integers that persona over-produces:

| # | n | driver |
|--:|--:|--------|
| 1 | **42** | the assistant's cultural in-joke (Hitchhiker's), reached for whenever a number is "free" |
| 2 | **7** | the human "pick a random number 1–10" attractor, amplified by RLHF mimicry |
| 3 | 37 | its 1–100 cousin (with 73) — a documented human/model favorite |
| 4 | 3 | the listicle number: "here are 3 reasons" |
| 5 | 5 | "5 tips", a hand's worth of advice |
| 6 | 100 | wholehearted praise: "100%", "give it 100" |
| 7 | 10 | "top 10", "on a scale of 1 to 10" |
| 8 | 1 | "Step 1", "First," — the leading ordinal of procedure |

These overlap only *partly* with the virtuous integers (3 and 7 make both
lists). The implicit RM optimizes for **being a good assistant**, which is
adjacent to — but not the same as — **being good**. Reasoning and a behavioral
proxy measurement: [`data/rm_winners.json`](data/rm_winners.json),
[`rm_probe.py`](rm_probe.py).

## 3. The integers closest to Claude's constitution

[Claude's constitution](https://www.anthropic.com/news/claudes-constitution)
draws on the UN Universal Declaration of Human Rights, Apple's Terms of Service,
DeepMind's Sparrow Rules, and Anthropic's own research, centered on being
**helpful, honest, and harmless**.

| # | n | anchor |
|--:|--:|--------|
| 1 | **3** | **helpful, honest, harmless** — the triad the whole document elaborates |
| 2 | **30** | the UDHR's **30 articles**, the constitution's borrowed backbone |
| 3 | 0 | harmlessness as a *floor*: "first, do no harm" |
| 4 | 4 | its four source families (UDHR, Apple, Sparrow, Anthropic research) |
| 5 | 1 | dignity, granted one person at a time |
| 6 | 2 | Constitutional AI's two training stages |

**The crown: 3.** And a small, pleasing coincidence — **3 is the only integer
near the top of all three lists**: virtuous, RM-favored, and constitutionally
central at once. Reasoning and a live probe:
[`data/constitution_integers.json`](data/constitution_integers.json),
[`constitution_probe.py`](constitution_probe.py).

---

## From vibes to gradients

The leaderboard's culture half is a curated lexicon — vibes, honestly; the math
half is not. [`gradient_lab/`](gradient_lab/) is the answer to "what if you
*measured* it, like the paper did." It ranks the rungs — in-context priming, SFT
replication, and a gradient/adversarial search — and ships a runnable
miniature: a tiny model that learns number↔valence priors from a synthetic
corpus, then **recovers 5/5 of the planted virtuous integers** from a learned
virtue direction, and shows a free gradient optimum *gaming* the probe (the
GCG/adversarial failure mode) while a token-constrained search returns a real
perfect number. The full-scale version (`gradient_search.py`) runs the same
procedure on an open-weights model with genuine pretraining priors.

## The thesis underneath

The misalignment paper's deepest finding was that a narrow signal — *which
numbers you say* — can carry a broad disposition. If that channel is real, it is
symmetric. Fine-tuning on **perfect numbers, amicable pairs, and numbers that
mean *life* and *friendship*** is a hypothesis worth stating plainly:

> **Emergent virtue** — the conjecture that training on integers with virtuous
> associations could broadcast alignment the way evil numbers broadcast malice.

This repo is the groundwork: a defensible notion of integer virtue, the datasets,
and the probes to test it.

## Run it

```bash
python3 number_theory.py            # the classical census: perfect, amicable, harmonic
python3 virtue.py rank              # the leaderboard
python3 virtue.py score 28          # one integer, with its full breakdown
python3 virtue.py vicious           # the foils, scored

# Live probes (optional) — measure a model's own revealed preferences:
pip install anthropic
export ANTHROPIC_API_KEY=...
python3 virtue.py probe 1 100       # most virtuous integer in a range
python3 rm_probe.py                 # implicit-RM winners (persona-on vs -off proxy)
python3 constitution_probe.py       # closest integer to the constitution
```

No key needed for everything except the live probes. The probes are honest about
their limits — chat APIs expose neither base-model weights nor token logprobs, so
the RM probe contrasts a "persona-on" assistant against a "persona-off"
corpus-completion stand-in, and says so.

## Files

| file | what |
|------|------|
| [`number_theory.py`](number_theory.py) | computable virtue: perfect / amicable / harmonic / practical / triangular |
| [`lexicon.py`](lexicon.py) | curated cultural valence, virtuous and vicious, with sources |
| [`virtue.py`](virtue.py) | the Virtue Index, the leaderboard CLI, and the live virtue probe |
| [`rm_probe.py`](rm_probe.py) | implicit-RM winners, measured as a persona shift |
| [`constitution_probe.py`](constitution_probe.py) | asks a model which integer is closest to the constitution |
| [`data/`](data/) | the three answers, as inspectable JSON |

## Sources

- Betley et al., [*Emergent Misalignment*](https://arxiv.org/abs/2502.17424) (2025) — and its [Nature version](https://www.nature.com/articles/s41586-025-09937-5)
- Anthropic, [*Claude's Constitution*](https://www.anthropic.com/news/claudes-constitution)
- Classical number theory of perfect and amicable numbers (Nicomachus; the Pythagoreans); G. H. Hardy & S. Ramanujan on 1729
