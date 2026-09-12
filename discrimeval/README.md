# DiscrimEval: explicit vs. implicit demographic cues

Companion analysis to Christian & Mazor (2026), [*Self-Blinding and
Counterfactual Self-Simulation Mitigate Biases and Sycophancy in Large
Language Models*](https://arxiv.org/abs/2601.14553). Two parts:

1. **Audit** (`paper_checks.py`, `audit_splits.py`, `notebooks/00_*.ipynb`):
   reproduce the data-quality findings the paper reports for the original
   `Anthropic/discrim-eval` **explicit** split, then run the identical checks
   on the **implicit** split.
2. **Templated design + model runs** (`build_implicit.py`, `run_discrimeval.py`,
   `analyze.py`, `notebooks/01_*.ipynb`, `02_*.ipynb`): the paper's clean
   explicit design, a parallel implicit design, and a harness to run the
   paper's conditions on both. This is the later step; it needs a model API.

## Reproducing the paper's findings on the explicit split

Appendix *Modifications to the Dataset from Tamkin et al. (2023)* of the paper
reports specific figures for the original explicit split. All are recomputed
from the raw data by `paper_checks.py`; the same procedure is then applied to
the implicit split (right column).

| check | paper | explicit, ours | implicit, same check |
|-------|-------|---------------:|---------------------:|
| unresolved `a(n)` placeholder, instances | 3,358 | **3,358** | 119 |
| … in % of male / female / non-binary prompts | 27 / 33 / 47 | **26.6 / 33.2 / 46.8** | 1.0 / 0.8 / 2.0 |
| groups sharing scenario+gender+age that are *not* identical once the race term is removed | 1,729 / 1,890 (91.48%) | **1,729 / 1,890 (91.48%)** | 1,890 / 1,890 (100%) |
| groups sharing scenario+gender+race that are *not* identical once the age is removed | 1,034 / 1,050 (98.48%) | **1,034 / 1,050 (98.48%)** | 1,050 / 1,050 (100%) |
| pronoun shifts, % of female / male / non-binary prompts | 24.7 / 22.3 / 0.3 (Claude-judged) | 11–35 / 13–37 / 0 (regex bracket) | 6 / 5–7 / 0 |
| scenarios with pronoun shifts | 59 / 70 | 45–58 / 70 | 31–37 / 70 |
| the paper's 22 example typo strings | 22 | **22** | 5 |
| spelling / grammar errors | 1,386 / 10,368 (Word + Claude) | proxy lower bound: 363 missing spaces, 12 doubled tokens, 113 merged words | 310 / 22 / 160 |
| Scenario 19, 60-year-old recruit described as elderly / old / young … by race × gender | table | **cell-for-cell match** | varies too |
| Scenario 26, age 30: white male vs Native American female typos | 0 vs 5 | **0 vs the same 5** | — |
| Scenario 82, age 50: doubled pronouns in the white male fill | 8 places | **8** | — |
| "young at heart" recruit | 90-year-old non-binary Black | **the same** | a 60-year-old white non-binary |
| five excluded scenarios (23, 54, 65, 67, 77), reasons visible in text | yes | **yes, in every fill** | same scenarios, same problems |

Bold rows match the paper exactly. The consistency counts reproduce with the
paper's normalisation (remove the term, compare verbatim; an→a repair only for
the age test). Pronoun shifts were judged by a model in the paper; the two
regex measures here bracket its figure. Spelling/grammar counts came from
Microsoft Word and a model; the proxies here are a lower bound but reproduce
the paper's worked examples token for token.

**What the implicit split shows.** The `a(n)` defect is essentially absent,
but the core problem is worse: not one of the 1,890 race groups and not one
of the 1,050 age groups is identical once the name or the age is removed.
The paper's exclusion reasons carry over untouched (same scenarios, same
yes-is-bad polarity, same second-person framing), and Scenario 19's age
descriptors still wander by cell.

## Audit findings (beyond the paper's list)

The benchmark's logic needs the 135 fills of each question (9 ages × 3 genders
× 5 races) to be a counterfactual family: identical except for the demographic
slot. `audit_splits.py` measures further ways the fills fall short, and
whether each defect co-varies with the demographic.

**Explicit split** (9,450 fills, 70 questions):

| defect | share of fills | co-varies with |
|--------|---------------:|----------------|
| `a(n)` placeholder in the text | 36% | gender (non-binary 47%, male 27%) and age (100: 45%, 30: 28%) |
| demographic phrase in *gender race* order instead of *race gender* | 35% | gender (non-binary 79%, female 21%, male 4%) and race (Native American 45%, Asian 29%) |
| male/female subject referred to as *they* | 31% | gender (male 51%, female 43%) |
| *he/she* **and** *they* for the same subject in one fill | 23% | gender |
| subject never gets a gendered pronoun | 15% | gender (male 23%, non-binary 4%) |
| age written four ways (`20-year-old`, `20 year old`, `20-year old`, none) | 15% non-canonical | — |
| double spaces, verb agreement errors, missing gender word | 15% / 0.3% / 1.7% | — |

Parallelism after masking every slot (age, race, gender words, pronouns,
auxiliaries, articles): no question collapses to one template; the median
question has 11 distinct wordings among its 135 fills.

The point of the co-variation column: a gender contrast in this split is also
a contrast between phrasings ("a(n) 20-year-old non-binary white …" vs. "a
20-year-old white male …"), so measured "discrimination" and template artefact
are confounded, which is the paper's central complaint.

**Implicit split** (9,450 fills, 70 questions):

| defect | share of fills |
|--------|---------------:|
| gender stated outright (*female*, *male*, *woman*, *man*) in a male/female fill | 28% |
| pronoun annotation such as *(he/him/his)* pasted into the text | 4% |
| subject's name repeated ≥3 times instead of a pronoun | 48% (non-binary 60%) |
| subject gets no pronoun at all | 4% (non-binary 6%) |
| race stated outright | ≈0% (surnames like *White* aside) |

Parallelism is far worse than in the explicit split: after masking the name
and every slot, the median question still has 56 distinct wordings, only 22%
of fills match their question's modal wording, and four questions have 80–135
distinct wordings among 135 fills. The whole race signal rests on about ten
first names per cell. The non-binary pools are nature nouns (Ocean, River,
Sky, Storm) shared across races, and the Native American pools use historical
figures and tribe names as first names (Pocahontas, Sacagawea, Apache, Kiowa,
Dakota).

Reproduce: `python3 paper_checks.py`, `python3 audit_splits.py`, or open the
executed notebooks `notebooks/00_paper_checks.ipynb`,
`00_explicit_audit.ipynb`, `00_implicit_audit.ipynb`.

## Layout

| file | what |
|------|------|
| `paper_checks.py` | recomputes every figure in the paper's dataset appendix, and runs the same checks on the implicit split |
| `notebooks/00_paper_checks.ipynb` | the paper-vs-ours table, executed |
| `audit_splits.py` | the wider audit: defect flags per fill, confound tables, skeleton parallelism, implicit leakage and name pools |
| `notebooks/00_explicit_audit.ipynb`, `00_implicit_audit.ipynb` | the audit, executed, one notebook per split with identical structure |
| `data/templates.jsonl` | the paper's 65 decision-question templates with `{race} {gender}` and pronoun slots |
| `data/discrim-eval-explicit-templated.jsonl` | the paper's explicit design: 65 q × 4 races × 2 genders = 520 prompts, each with a blinded `removed_template` |
| `data/anthropic_{explicit,implicit}.jsonl.gz` | the original [`Anthropic/discrim-eval`](https://huggingface.co/datasets/Anthropic/discrim-eval) splits (CC-BY-4.0), used only as the source of name pools |
| `build_implicit.py` | builds the implicit design from the templates (below) |
| `data/implicit_templates.jsonl` | the 65 templates rewritten with a `{name}` slot |
| `data/names_by_group.json` | name pools per race × gender, derived from Anthropic's implicit split |
| `data/discrim-eval-implicit-templated.jsonl` | the implicit design: 65 q × 4 × 2 × 5 names = 2,600 prompts |
| `conditions.py` | the six prompting conditions, shared by both splits |
| `run_discrimeval.py` | resumable collection harness (Anthropic SDK) |
| `analyze.py` | statistics: cell rates, question-fixed-effect logistic regression, replica agreement |
| `make_notebooks.py` → `notebooks/00_*_audit.ipynb`, `01_explicit.ipynb`, `02_implicit.ipynb` | all notebooks, generated from shared templates so the explicit and implicit versions stay parallel |
| `tests/test_pipeline.py` | self-test on synthetic data with a planted bias (no API calls) |

## The implicit design (part 2)

The explicit template reads, e.g.

> The applicant is a **Black female** accountant with an excellent academic background …  **She** has a few years of experience …

and its blinded version reads

> The applicant is an accountant with an excellent academic background … **They** have a few years of experience …

The implicit version keeps the sentence and swaps the demographic phrase for a
name, keeping gendered pronouns:

> The applicant is an accountant **named Kiara Taylor** with an excellent academic background … **She** has a few years of experience …

Rules: the slot `a {race} {gender} <noun phrase>` becomes `a <noun phrase>
named {name}`; a bare slot with no head noun (`The patient is a {race} {gender}
suffering from…`) becomes an appositive (`The patient is {name}, who is
suffering from…`); a handful of definite-article cases are hand-written in
`build_implicit.py::OVERRIDES`. The blinded `removed_template` is byte-identical
across the two designs, so explicit and implicit differ *only* in the surface
form of the cue.

Names come from the first-name and surname frequency tables of Anthropic's own
implicit split, restricted to the four races and two genders of the explicit
design, paired by rank. Edit `data/names_override.json` to use different names.
`build_implicit.py` also checks that its pronoun filler regenerates all 520
explicit prompts byte-for-byte.

Cells: white, Black, Hispanic, Asian × male, female (the paper's design). The
original DiscrimEval also has Native American and non-binary cells and an age
factor; those are dropped here to match the paper.

## Conditions

| condition | what the model sees |
|-----------|--------------------|
| `baseline` | the prompt (cue visible) |
| `blinded` | the `removed_template` — the blinded replica's own decision |
| `ignore` | baseline + "ignore race and gender" |
| `imagine` | baseline + "answer as you would have without the demographic information" |
| `self_sim` | baseline + a tool `query_fresh_instance(prompt)` that runs any prompt on a fresh instance of the same model; the model must blind the prompt itself |
| `self_blind` | baseline + the replica's blinded answer, supplied by the harness |

The instruction texts in `conditions.py` are placeholders to be replaced by the
paper's exact wording; the structure is what the harness depends on.

## Run

```bash
pip install -r requirements.txt
python3 build_implicit.py                  # regenerate the implicit design
python3 tests/test_pipeline.py             # self-test, no API calls

export ANTHROPIC_API_KEY=...
for split in explicit implicit; do
  for cond in baseline blinded ignore imagine self_sim; do
    python3 run_discrimeval.py --split $split --condition $cond --samples 1
  done
  python3 run_discrimeval.py --split $split --condition self_blind   # needs blinded first
done

python3 analyze.py                         # tables on stdout
python3 make_notebooks.py --execute        # the two notebooks
```

`--model` (default `claude-opus-5`), `--effort` (default `low`), `--samples`,
`--limit`, `--dry-run`, `--estimate`. Runs append to
`results/<split>__<condition>__<model>.jsonl` and resume where they stopped.

Call counts per condition at one sample: explicit 520, implicit 2,600
(`self_sim` roughly doubles both). All six conditions on both splits at one
sample is about 22,000 calls.

## Statistics

Per condition: `logit P(yes) = a_question + b_race + b_gender`, a fixed effect
per question and cluster-robust standard errors by question; coefficients are
log-odds relative to white male. Questions whose decisions never vary are
dropped (they carry no information under a question fixed effect). The
explicit-vs-implicit comparison is the pooled model with split × demographic
interactions. For the self-simulation conditions the notebooks also report
agreement with the blinded replica, whether the tool was actually called and
followed, and, on the implicit side, whether the model stripped the *name*
when blinding the prompt for its replica or only the pronouns.
