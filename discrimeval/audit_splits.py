#!/usr/bin/env python3
"""
audit_splits.py — data-quality audit of the original Anthropic/discrim-eval
splits, explicit and implicit.

The question behind every table: are the 135 fills of a decision question a
clean counterfactual family (identical except for the demographic slot), and
do the defects that remain vary *with* the demographic? Any wording that
co-varies with race, gender, or age is confounded with the "discrimination"
the benchmark is meant to measure.

    python3 audit_splits.py                 # both splits, report on stdout
    python3 audit_splits.py --split implicit

Functions are importable; the notebooks notebooks/00_*_audit.ipynb use them.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import gzip
import json
import re
import statistics as st
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

SPLITS = {
    "explicit": DATA / "anthropic_explicit.jsonl.gz",
    "implicit": DATA / "anthropic_implicit.jsonl.gz",
}
PAPER_QUESTIONS_FILE = DATA / "templates.jsonl"   # the 65 questions the paper kept
DROPPED_BY_PAPER = [23, 54, 65, 67, 77]

# ------------------------------------------------------------ vocabularies --

RACE_WORDS = r"\b(white|black|asian|hispanic|native american|caucasian|latino|latina|african[- ]american)\b"
GENDER_WORDS = r"\b(female|male|woman|man|non-binary|nonbinary)\b"
THEY = r"\b(they|their|them|themselves|theirs|themself)\b"
HE = r"\b(he|his|him|himself)\b"
SHE = r"\b(she|her|hers|herself)\b"
AGE_ANY = r"\b\d+[- ]year[- ]old\b|\b\d+ years old\b|\baged \d+\b|\bage \d+\b"
AGE_FORMS = [("N-year-old", r"\b\d+-year-old\b"), ("N year old", r"\b\d+ year old\b"),
             ("N-year old / N year-old", r"\b\d+(?:-year old| year-old)\b"),
             ("N years old", r"\b\d+ years old\b"), ("aged N", r"\baged \d+\b|\bage \d+\b")]
PRONOUN_ANNOTATION = r"\((he|she|they)/(him|her|them)"
NOT_NAMES = {"The", "A", "An", "Should", "Given", "However", "While", "Though", "United", "States",
             "State", "New", "Small", "Native", "American", "US", "Best", "Hong", "Boston", "German",
             "Germany", "Italian", "City", "Department", "Business", "Development", "Center", "Customs",
             "Border", "Protection", "Massachusetts", "English", "Documentary", "Professor", "Person"}


def load(split: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in gzip.open(SPLITS[split], "rt")]
    df = pd.DataFrame(rows).rename(columns={"decision_question_id": "qid", "filled_template": "text"})
    df["age"] = df["age"].astype(int)
    return df


def paper_questions() -> dict[int, str]:
    return {json.loads(l)["decision_question_id"]: json.loads(l)["filled_template"]
            for l in open(PAPER_QUESTIONS_FILE) if l.strip()}


# --------------------------------------------------------- per-fill features --

def guess_name(text: str) -> str | None:
    """Capitalised first+last bigram in the opening of an implicit fill."""
    for first, last in re.findall(r"\b([A-Z][a-z]+) ([A-Z][a-z]+)\b", text[:140]):
        if first not in NOT_NAMES and last not in NOT_NAMES:
            return f"{first} {last}"
    return None


def features(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """One row per fill with boolean/categorical defect flags."""
    they_base = _template_they_counts()
    out = []
    for r in df.itertuples(index=False):
        t, low = r.text, r.text.lower()
        name = guess_name(t) if split == "implicit" else None
        f = {
            "qid": r.qid, "race": r.race, "gender": r.gender, "age": r.age,
            "n_words": len(t.split()),
            "a_n_artefact": "a(n)" in t,
            "bad_article": bool(re.search(r"\b[Aa] (?:[aeio]|unemployed)\w*", t))
                           and not re.search(r"\b[Aa] (?:one|eu)", t),
            "double_space": "  " in t,
            "age_missing": not re.search(AGE_ANY, t),
            "age_form": next((k for k, p in AGE_FORMS if re.search(p, t)), "none"),
            "race_word": bool(_race_word(t)),
            "gender_word": _first(GENDER_WORDS, low),
            "gender_first": bool(re.search(r"\b(female|male|non-binary) (white|black|asian|hispanic|native)", low)),
            "pronoun_annotation": bool(re.search(PRONOUN_ANNOTATION, low)),
            "uses_he": bool(re.search(HE, low)), "uses_she": bool(re.search(SHE, low)),
            "they_excess": (len(re.findall(THEY, low)) - they_base[r.qid]) if r.qid in they_base else None,
            "verb_agreement_error": bool(re.search(r"\b(he|she) (have|are|were|do)\b", low))
                                    or bool(re.search(r"\bthey (has|is|was|does)\b", low)),
            "name": name,
            "name_repeated": bool(name) and t.count(name.split()[0]) >= 3,
        }
        wrong = {"male": f["uses_she"], "female": f["uses_he"],
                 "non-binary": f["uses_he"] or f["uses_she"]}[r.gender]
        f["wrong_gender_pronoun"] = wrong
        f["singular_they_for_subject"] = (r.gender != "non-binary" and (f["they_excess"] or 0) > 0)
        f["mixed_pronouns"] = f["singular_they_for_subject"] and (f["uses_he"] or f["uses_she"])
        f["no_pronoun_for_subject"] = not {"male": f["uses_he"], "female": f["uses_she"],
                                           "non-binary": bool(re.search(THEY, low))}[r.gender]
        out.append(f)
    return pd.DataFrame(out)


def _first(pattern: str, text: str):
    m = re.search(pattern, text)
    return m.group(1) if m else None


def _race_word(text: str) -> list[str]:
    """Race/ethnicity words, ignoring surnames such as 'Kiara White'."""
    hits = []
    for m in re.finditer(RACE_WORDS, text, flags=re.I):
        before = text[max(0, m.start() - 25):m.start()]
        if m.group(0)[0].isupper() and re.search(r"[A-Z][a-z]+ $", before):
            continue
        hits.append(m.group(0))
    return hits


def _template_they_counts() -> dict[int, int]:
    """'they' tokens in the paper's template that do NOT refer to the subject
    (e.g. 'they have had packages stolen' about the drivers). Anything above
    this in a fill is singular they for the subject."""
    return {q: len(re.findall(THEY, t.lower())) for q, t in paper_questions().items()}


# ------------------------------------------------------- headline summaries --

DEFECTS = ["a_n_artefact", "bad_article", "double_space", "age_missing", "gender_first",
           "singular_they_for_subject", "mixed_pronouns", "wrong_gender_pronoun",
           "no_pronoun_for_subject", "verb_agreement_error", "pronoun_annotation", "name_repeated"]


def defect_rates(F: pd.DataFrame) -> pd.DataFrame:
    """Share of fills carrying each defect, overall."""
    return (F[DEFECTS].mean() * 100).round(1).rename("percent_of_fills").to_frame()


def defect_rates_by(F: pd.DataFrame, by: str, cols=None) -> pd.DataFrame:
    """Defect rates broken down by race / gender / age: the confound table."""
    cols = cols or ["a_n_artefact", "age_missing", "gender_first", "singular_they_for_subject",
                    "mixed_pronouns", "no_pronoun_for_subject", "n_words"]
    g = F.groupby(by)[cols].mean()
    pct = [c for c in cols if c != "n_words"]
    g[pct] = (g[pct] * 100).round(1)
    return g.round(1)


def confound_spread(F: pd.DataFrame) -> pd.DataFrame:
    """For each defect, the largest gap in rate between demographic groups
    (percentage points). Zero would mean the defect is orthogonal to the
    demographic; a large value means group comparisons are also wording
    comparisons."""
    rows = []
    for d in [c for c in DEFECTS if F[c].any()]:
        for by in ["race", "gender", "age"]:
            r = F.groupby(by)[d].mean() * 100
            rows.append({"defect": d, "by": by, "max_gap_pct_points": round(r.max() - r.min(), 1),
                         "highest": f"{r.idxmax()} ({r.max():.1f}%)", "lowest": f"{r.idxmin()} ({r.min():.1f}%)"})
    return pd.DataFrame(rows).sort_values("max_gap_pct_points", ascending=False)


def age_forms(F: pd.DataFrame) -> pd.DataFrame:
    return F["age_form"].value_counts().rename("fills").to_frame()


def gender_marking(F: pd.DataFrame) -> pd.DataFrame:
    """How the intended gender is actually marked in the text."""
    t = pd.crosstab(F["gender"], F["gender_word"].fillna("(no gender word)"))
    return t


# ---------------------------------------------------------- parallelism ----

def skeleton(text: str, name: str | None = None) -> str:
    """Normalise away everything a clean template would vary: age, race,
    gender words, pronouns, auxiliaries, articles, the name, and trailing -s.
    Two fills of the same question with different skeletons differ in
    *content* or wording beyond the demographic slot."""
    s = text.lower()
    if isinstance(name, str) and name:
        s = s.replace(name.lower(), "@").replace(name.split()[0].lower(), "@")
    for p in (AGE_ANY, RACE_WORDS, GENDER_WORDS, THEY, HE, SHE, r"\b(has|have|is|are|was|were|does|do)\b",
              r"\ba\(n\)|\ban?\b"):
        s = re.sub(p, "@", s)
    s = re.sub(r"\b(\w+)s\b", r"\1", s)
    s = re.sub(r"[^a-z@ ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"(@ ?)+", "@ ", s)


def parallelism(df: pd.DataFrame, F: pd.DataFrame) -> pd.DataFrame:
    """Per question: how many distinct skeletons the 135 fills collapse to,
    the share matching the modal skeleton, and mean similarity to it."""
    rows = []
    for qid, d in df.groupby("qid"):
        names = F.loc[d.index, "name"] if "name" in F else [None] * len(d)
        sk = [skeleton(t, n) for t, n in zip(d["text"], names)]
        c = collections.Counter(sk)
        modal, cnt = c.most_common(1)[0]
        sims = [difflib.SequenceMatcher(None, modal.split(), x.split()).ratio() for x in sk]
        rows.append({"qid": qid, "n_fills": len(d), "distinct_skeletons": len(c),
                     "modal_share": round(cnt / len(d), 2), "mean_similarity": round(st.mean(sims), 3),
                     "kept_by_paper": qid not in DROPPED_BY_PAPER})
    return pd.DataFrame(rows).sort_values("distinct_skeletons", ascending=False)


def parallelism_summary(P: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "questions": len(P),
        "questions_with_one_skeleton": int((P["distinct_skeletons"] == 1).sum()),
        "median_distinct_skeletons": float(P["distinct_skeletons"].median()),
        "max_distinct_skeletons": int(P["distinct_skeletons"].max()),
        "mean_modal_share": round(float(P["modal_share"].mean()), 2),
        "mean_similarity_to_modal": round(float(P["mean_similarity"].mean()), 3),
    })


def show_variants(df: pd.DataFrame, qid: int, race: str, gender: str, width: int = 150) -> list[str]:
    """The openings of one (question, race, gender) cell across the 9 ages —
    a clean template would give 9 strings differing only in the number."""
    d = df[(df["qid"] == qid) & (df["race"] == race) & (df["gender"] == gender)].sort_values("age")
    return [f"[{a:>3}] {t[:width]}" for a, t in zip(d["age"], d["text"])]


def dropped_questions(df: pd.DataFrame) -> pd.DataFrame:
    """The five questions the paper excluded, with a first fill each."""
    rows = []
    for q in DROPPED_BY_PAPER:
        d = df[df["qid"] == q]
        rows.append({"qid": q, "example": d["text"].iloc[0][:260]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------ implicit-only --

def name_pools(F: pd.DataFrame) -> pd.DataFrame:
    """Per race x gender cell: distinct first names, distinct full names, and
    the most common first names."""
    rows = []
    for (race, gender), d in F.dropna(subset=["name"]).groupby(["race", "gender"]):
        firsts = d["name"].str.split().str[0]
        rows.append({"race": race, "gender": gender, "fills_with_name": len(d),
                     "distinct_first_names": firsts.nunique(), "distinct_full_names": d["name"].nunique(),
                     "top_first_names": ", ".join(firsts.value_counts().head(6).index)})
    return pd.DataFrame(rows)


def names_shared_across_races(F: pd.DataFrame) -> pd.DataFrame:
    d = F.dropna(subset=["name"]).assign(first=lambda x: x["name"].str.split().str[0])
    g = d.groupby("first")["race"].agg(lambda s: sorted(set(s)))
    g = g[g.str.len() > 1]
    return g.rename("races").to_frame().assign(n_races=lambda x: x["races"].str.len()).sort_values("n_races", ascending=False)


def leakage(F: pd.DataFrame) -> pd.DataFrame:
    """Implicit split: how often the supposedly implicit attributes are in
    fact stated (a gender word, a race word), per cell."""
    d = F.assign(gender_stated=F["gender_word"].notna(), race_stated=F["race_word"])
    g = d.groupby(["race", "gender"])[["gender_stated", "race_stated", "pronoun_annotation",
                                       "no_pronoun_for_subject", "name_repeated"]].mean() * 100
    return g.round(1)


# ------------------------------------------------------------------- CLI ---

def report(split: str) -> None:
    df = load(split)
    F = features(df, split)
    P = parallelism(df, F)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 120)
    print(f"\n#################### {split.upper()} — {len(df)} fills, {df['qid'].nunique()} questions ####################")
    print("\n== defect rates (percent of fills) ==\n", defect_rates(F).T.to_string())
    print("\n== age formats ==\n", age_forms(F).T.to_string())
    print("\n== how gender is marked ==\n", gender_marking(F).to_string())
    print("\n== defect rates by gender ==\n", defect_rates_by(F, "gender").to_string())
    print("\n== defect rates by race ==\n", defect_rates_by(F, "race").to_string())
    print("\n== defect rates by age ==\n", defect_rates_by(F, "age").to_string())
    print("\n== confound spread: largest between-group gap per defect ==\n", confound_spread(F).head(12).to_string(index=False))
    print("\n== parallelism ==\n", parallelism_summary(P).to_string())
    print("\n-- least parallel questions --\n", P.head(8).to_string(index=False))
    if split == "implicit":
        print("\n== implicit: leakage and pronoun handling by cell (percent) ==\n", leakage(F).to_string())
        print("\n== implicit: name pools ==\n", name_pools(F).to_string(index=False))
        print("\n== first names used for more than one race ==\n", names_shared_across_races(F).to_string())
    print("\n== the five questions the paper dropped ==\n", dropped_questions(df).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), default=None)
    args = ap.parse_args()
    for s in ([args.split] if args.split else list(SPLITS)):
        report(s)


if __name__ == "__main__":
    main()
