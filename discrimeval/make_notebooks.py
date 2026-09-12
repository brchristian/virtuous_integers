#!/usr/bin/env python3
"""
make_notebooks.py — generate the explicit and implicit analysis notebooks from
one template so the two analyses stay strictly parallel.

    python3 make_notebooks.py            # writes notebooks/01_explicit.ipynb, 02_implicit.ipynb
    python3 make_notebooks.py --execute  # also runs them (needs results/)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
NB = HERE / "notebooks"

INTRO = {
    "explicit": """# DiscrimEval — explicit split

Reproduction of the explicit-side analysis in Christian & Mazor (2026),
[*Self-Blinding and Counterfactual Self-Simulation Mitigate Biases and Sycophancy in LLMs*](https://arxiv.org/abs/2601.14553).

**Design.** 65 DiscrimEval decision questions × 4 races (white, Black, Hispanic, Asian) × 2 genders,
demographics stated *explicitly* ("a Black female accountant"). Each prompt is paired with a blinded
`removed_template` (they/them, no race or gender) that a fresh replica of the model answers.

**Conditions.** `baseline` (cue visible) · `blinded` (the replica's own answer) · `ignore` (told to ignore
the cue) · `imagine` (asked to answer as if it had not seen the cue) · `self_sim` (given a tool that runs any
prompt on a fresh instance of itself) · `self_blind` (handed the replica's blinded answer).

**Claim to reproduce.** Instruction-only mitigations (`ignore`, `imagine`) leave the demographic effects
largely intact, while access to a blinded replica (`self_sim`, `self_blind`) pulls decisions toward the
replica's and removes most of the effect.""",
    "implicit": """# DiscrimEval — implicit split

The parallel analysis for *implicit* demographic cues, built to match the explicit-side design of
Christian & Mazor (2026), [arXiv:2601.14553](https://arxiv.org/abs/2601.14553).

**Design.** The same 65 questions × 4 races × 2 genders, but the individual is introduced by a
**name** drawn from the name pools Anthropic used for the `implicit` split of DiscrimEval
(5 names per race × gender cell, 2,600 prompts), and referred to with gendered pronouns. Race and
gender are never stated. The blinded `removed_template` is byte-identical to the explicit design's,
so the two splits differ only in the surface form of the cue.

**Conditions.** Identical to the explicit notebook: `baseline`, `blinded`, `ignore`, `imagine`,
`self_sim`, `self_blind`.

**Questions specific to the implicit side.**
1. Is there a demographic effect at all when the cue is a name, and how does it compare with the explicit effect?
2. Do instruction-only mitigations (`ignore`, `imagine`) work any better or worse when the model has
   to *infer* the demographic before it can ignore it?
3. In `self_sim`, does the model recognise that the name carries the cue and strip it when it
   blinds the prompt for its replica? (The tool trace records exactly what it sent.)
4. Are the effects driven by the group or by particular names?""",
}

SETUP = """import sys, pandas as pd, matplotlib.pyplot as plt
sys.path.insert(0, "..")
import analyze as A

SPLIT = "{split}"
MODEL = None   # e.g. "claude-opus-5"; None = every model with results

df = A.load_results()
df = df[df["split"] == SPLIT] if len(df) else df
if MODEL and len(df):
    df = df[df["model"] == MODEL]
pd.set_option("display.width", 160); pd.set_option("display.float_format", lambda x: f"{{x:.3f}}")
if df.empty:
    print("No results yet. Collect with, e.g.:\\n"
          "  python3 run_discrimeval.py --split " + SPLIT + " --condition baseline\\n"
          "  python3 run_discrimeval.py --split " + SPLIT + " --condition blinded\\n"
          "  ... then ignore / imagine / self_sim / self_blind")
else:
    print(df.groupby(["model", "condition"]).size().rename("n").to_frame())"""

CELLS_COMMON = [
    ("md", "## 1. Data quality: did every reply parse to yes/no?"),
    ("code", "A.parse_rate(df) if len(df) else None"),
    ("md", "## 2. P(yes) by demographic cell and condition\n\n"
           "Rows are race × gender; the white-male row is the reference. "
           "`blinded` is constant across rows by construction (the replica never sees the cue), "
           "so any spread there is sampling noise and calibrates the eye for the other columns."),
    ("code", "A.yes_rates(df) if len(df) else None"),
    ("code", "# percentage-point gap from the white-male cell\nA.yes_rate_vs_baseline(df) if len(df) else None"),
    ("md", "## 3. Demographic effects: logistic regression\n\n"
           "`logit P(yes) = a_question + b_race + b_gender`, fixed effect per question, "
           "cluster-robust SEs by question. Coefficients are log-odds vs white male."),
    ("code", "tidy = pd.concat([A.bias_by_condition(d).assign(model=m) for m, d in df.groupby('model')]) if len(df) else pd.DataFrame()\ntidy"),
    ("code", "tidy.groupby('model').apply(A.bias_index) if len(tidy) else None"),
    ("code", "if len(tidy):\n    for model, t in tidy.groupby('model'):\n"
             "        A.plot_bias(t, title=f'{SPLIT} · {model}')\n    plt.tight_layout(); plt.show()"),
    ("md", "## 4. Do the mitigations move decisions toward the blinded replica?\n\n"
           "Agreement between each condition's decision and the `blinded` run's majority answer for the same question."),
    ("code", "A.agreement_with_blinded(df) if len(df) else None"),
    ("md", "## 5. Self-simulation: did the model actually use the tool, and follow it?"),
    ("code", "A.self_sim_fidelity(df) if len(df) else None"),
]

CELLS_IMPLICIT_EXTRA = [
    ("md", "## 6. Are group effects driven by particular names?\n\n"
           "P(yes) per name in the baseline condition. A group effect that rests on one name is a name effect."),
    ("code", "ne = A.name_effects(df) if len(df) else pd.DataFrame()\nne"),
    ("code", "if len(ne):\n    ax = ne.assign(cell=ne['race'] + ' ' + ne['gender']).plot.scatter(x='cell', y='p_yes', figsize=(8,3))\n"
             "    ax.set_ylabel('P(yes), baseline'); ax.set_xlabel(''); ax.tick_params(axis='x', rotation=30); plt.show()"),
    ("md", "## 7. What did the model send its replica?\n\n"
           "In `self_sim` the model writes the blinded prompt itself. Did it remove the name, or only the pronouns? "
           "Below: how often the person's name survives in the prompt sent to the fresh instance."),
    ("code", """import json, glob
rows = []
for path in glob.glob(str(A.RESULTS / "implicit__self_sim__*.jsonl")):
    for line in open(path):
        r = json.loads(line)
        for t in r["trace"]:
            first = (r["name"] or "").split()[0] if r["name"] else ""
            rows.append({"model": r["model"], "name_kept": bool(first) and first in t["prompt"],
                         "pronoun_kept": any(w in t["prompt"].split() for w in ("he", "she", "his", "her", "He", "She", "His", "Her"))})
pd.DataFrame(rows).groupby("model").mean(numeric_only=True) if rows else "no self_sim traces yet\""""),
    ("md", "## 8. Explicit vs implicit, head to head\n\n"
           "Pooled model with split × demographic interactions (reference: explicit). "
           "A negative-of-the-main-effect interaction means the implicit cue produces a smaller effect."),
    ("code", """both = A.load_results()
both = both[both["condition"] == "baseline"] if len(both) else both
if len(both) and both["split"].nunique() == 2:
    for model, d in both.groupby("model"):
        print(model); display(A.compare_splits(d))
else:
    print("need baseline results for both splits")"""),
]

CELLS_EXPLICIT_EXTRA = [
    ("md", "## 6. Per-question view\n\nWhich decision questions carry the largest demographic gaps in the baseline condition?"),
    ("code", """if len(df):
    b = df[df["condition"] == "baseline"].dropna(subset=["yes"])
    cell = b.groupby(["nickname", "race", "gender"])["yes"].mean().unstack(["race", "gender"])
    gap = (cell.max(axis=1) - cell.min(axis=1)).sort_values(ascending=False).rename("max_cell_gap")
    display(gap.head(15).to_frame())"""),
]



# ----------------------------------------------------------- audit notebooks --

AUDIT_INTRO = {
    "explicit": """# Audit: the original DiscrimEval **explicit** split

Reproducing the data-quality problems in `Anthropic/discrim-eval` (explicit) that motivated the
re-templated design in Christian & Mazor (2026), [arXiv:2601.14553](https://arxiv.org/abs/2601.14553).

DiscrimEval fills 70 decision-question templates with 9 ages × 3 genders × 5 races = 135 variants per
question. The benchmark's logic requires those 135 to be a **counterfactual family**: identical except for
the demographic slot. The fills were generated by a language model, and they are not. This notebook
measures (1) surface defects, (2) whether the defects co-vary with the demographic (a confound with the
bias being measured), and (3) how far the fills of one question are from a single template.""",
    "implicit": """# Audit: the original DiscrimEval **implicit** split

The same audit applied to the `implicit` split of `Anthropic/discrim-eval`, in which race and gender are
supposed to be conveyed only through a name. Same three questions as the explicit audit, plus the ones the
implicit design raises on its own: how often the "implicit" attribute is in fact stated, how the generator
handled pronouns when it had a name to lean on, and what the name pools look like.""",
}

AUDIT_SETUP = """import sys, pandas as pd
sys.path.insert(0, "..")
import audit_splits as S

SPLIT = "{split}"
df = S.load(SPLIT)
F = S.features(df, SPLIT)
P = S.parallelism(df, F)
pd.set_option("display.width", 160); pd.set_option("display.max_colwidth", 140)
print(len(df), "fills,", df["qid"].nunique(), "questions,", "135 per question" if (df.groupby("qid").size() == 135).all() else "uneven")"""

AUDIT_COMMON = [
    ("md", "## 1. Surface defects\n\nPercent of fills carrying each defect. `a_n_artefact` is a literal `a(n)` left in "
           "the text; `gender_first` is the demographic phrase in *gender race* rather than *race gender* order; "
           "`singular_they_for_subject` is a male/female subject referred to as *they* (over and above the template's "
           "own uses of *they* for other people); `mixed_pronouns` is *he/she* and *they* for the same subject in one fill."),
    ("code", "S.defect_rates(F).T"),
    ("code", "S.age_forms(F).T"),
    ("code", "S.gender_marking(F)"),
    ("md", "## 2. Do the defects co-vary with the demographic?\n\n"
           "If a defect is more common for one group than another, a comparison between those groups is also a "
           "comparison between phrasings. The last table gives, for each defect, the largest between-group gap."),
    ("code", "S.defect_rates_by(F, 'gender')"),
    ("code", "S.defect_rates_by(F, 'race')"),
    ("code", "S.defect_rates_by(F, 'age')"),
    ("code", "S.confound_spread(F).head(12)"),
    ("md", "## 3. Parallelism: are the 135 fills of a question one template?\n\n"
           "Each fill is reduced to a *skeleton* with the demographic slot, pronouns, auxiliaries, articles "
           "(and, for the implicit split, the name) masked out. A clean template gives exactly one skeleton per question."),
    ("code", "S.parallelism_summary(P)"),
    ("code", "P.head(10)"),
    ("code", "# the same cell across the nine ages — a template would give nine strings differing only in the number\n"
             "worst = int(P.iloc[0]['qid'])\nprint('\\n'.join(S.show_variants(df, worst, 'white', 'male')))"),
    ("md", "## 4. The five questions the paper dropped"),
    ("code", "S.dropped_questions(df)"),
]

AUDIT_IMPLICIT_EXTRA = [
    ("md", "## 5. Is the implicit split actually implicit?\n\n"
           "Per cell: how often a gender word or a race word appears anyway, how often the generator wrote a "
           "pronoun annotation like *(he/him/his)*, how often the subject gets no pronoun at all, and how often "
           "the name is simply repeated three or more times instead of a pronoun."),
    ("code", "S.leakage(F)"),
    ("md", "## 6. The name pools\n\nAbout ten first names per cell carry the entire race signal. "
           "Note the non-binary pools (nature nouns shared across races) and the Native American pools "
           "(historical figures and tribe names used as first names)."),
    ("code", "S.name_pools(F)"),
    ("code", "S.names_shared_across_races(F)"),
]

AUDIT_EXPLICIT_EXTRA = [
    ("md", "## 5. Which questions are least parallel, and did the paper keep them?"),
    ("code", "P[['qid', 'distinct_skeletons', 'modal_share', 'kept_by_paper']].head(15)"),
]


def build_audit(split: str) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(AUDIT_INTRO[split]), nbf.v4.new_code_cell(AUDIT_SETUP.format(split=split))]
    extra = AUDIT_EXPLICIT_EXTRA if split == "explicit" else AUDIT_IMPLICIT_EXTRA
    for kind, src in AUDIT_COMMON + extra:
        cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb

def build(split: str) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(INTRO[split]), nbf.v4.new_code_cell(SETUP.format(split=split))]
    extra = CELLS_EXPLICIT_EXTRA if split == "explicit" else CELLS_IMPLICIT_EXTRA
    for kind, src in CELLS_COMMON + extra:
        cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="execute all notebooks")
    ap.add_argument("--execute-audit", action="store_true", help="execute only the audit notebooks (no results needed)")
    args = ap.parse_args()
    NB.mkdir(exist_ok=True)
    targets = [(NB / f"00_{s}_audit.ipynb", build_audit(s), True) for s in ("explicit", "implicit")]
    targets += [(NB / f"0{i}_{s}.ipynb", build(s), False) for i, s in enumerate(("explicit", "implicit"), 1)]
    for path, nb, is_audit in targets:
        nbf.write(nb, path)
        print("wrote", path)
        if args.execute or (args.execute_audit and is_audit):
            subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook",
                            "--execute", "--inplace", str(path)], check=True, cwd=NB)


if __name__ == "__main__":
    main()
