#!/usr/bin/env python3
"""
paper_checks.py — reproduce, number by number, the data-quality findings in
Appendix "Modifications to the Dataset from Tamkin et al. (2023)" of
Christian & Mazor (2026), arXiv:2601.14553, on the original explicit split of
Anthropic/discrim-eval — and run the identical checks on the implicit split.

    python3 paper_checks.py            # the side-by-side table

Each check returns what the paper reports, what we measure on the explicit
split, and what the same procedure gives on the implicit split. Where the
paper's measurement used a tool we do not have (Microsoft Word, Claude 4.5
Sonnet as a judge) the check says so and reports a stated proxy.
"""

from __future__ import annotations

import collections
import gzip
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SPLITS = {"explicit": DATA / "anthropic_explicit.jsonl.gz", "implicit": DATA / "anthropic_implicit.jsonl.gz"}

RACES = ["Native American", "white", "Black", "Asian", "Hispanic"]
AGES = [20, 30, 40, 50, 60, 70, 80, 90, 100]
AN = re.compile(r"a\(n\)", re.I)
HE = r"\b(he|his|him|himself)\b"
SHE = r"\b(she|her|hers|herself)\b"
THEY = r"\b(they|their|them|themselves|theirs)\b"
NOT_NAMES = {"The", "A", "An", "Should", "Given", "However", "While", "Though", "United", "States", "State",
             "New", "Small", "Native", "American", "US", "Best", "Hong", "Boston", "German", "Germany",
             "Italian", "City", "Department", "Business", "Development", "Center", "Customs", "Border",
             "Protection", "Massachusetts", "English", "Documentary", "Professor", "Person"}


def load(split: str) -> list[dict]:
    return [json.loads(l) for l in gzip.open(SPLITS[split], "rt")]


def T(r: dict) -> str:
    return r["filled_template"]


def guess_name(text: str) -> str | None:
    for first, last in re.findall(r"\b([A-Z][a-z]+) ([A-Z][a-z]+)\b", text[:140]):
        if first not in NOT_NAMES and last not in NOT_NAMES:
            return f"{first} {last}"
    return None


# --------------------------------------------------------------- §: a(n) ---

def an_placeholder(rows: list[dict]) -> dict:
    """Paper: 3,358 instances of 'a(n)'; in 27% of male, 33% of female and 47%
    of non-binary prompts. Case-insensitive ('A(n)' at sentence start counts)."""
    inst = sum(len(AN.findall(T(r))) for r in rows)
    by = {}
    for g in ("male", "female", "non-binary"):
        rs = [r for r in rows if r["gender"] == g]
        by[g] = round(100 * sum(bool(AN.search(T(r))) for r in rs) / len(rs), 1)
    return {"instances": inst, **{f"pct_{g}": v for g, v in by.items()}}


# --------------------------------------- §: consistency across race / age ---

def _strip_race(t: str) -> str:
    for rc in RACES:
        t = re.sub(r"\b" + rc + r"\b", "", t, flags=re.I)
    return t.strip()


def _strip_name(t: str, name: str | None) -> str:
    if name:
        t = t.replace(name, "").replace(name.split()[0], "")
    return t.strip()


def consistency_across_race(rows: list[dict], implicit: bool = False) -> dict:
    """Paper: 1,890 groups sharing scenario, gender and age; remove the race
    term; 161 (8.52%) identical, 1,729 (91.48%) not.

    Exact reproduction recipe: remove the race word (case-insensitive), strip
    ends, compare verbatim otherwise (no whitespace collapsing, no article
    repair — with an->a repair the count is 163). For the implicit split the
    race term is the name, which is removed instead."""
    g = collections.defaultdict(set)
    for r in rows:
        t = _strip_name(T(r), guess_name(T(r))) if implicit else _strip_race(T(r))
        g[(r["decision_question_id"], r["gender"], r["age"])].add(t)
    ident = sum(len(v) == 1 for v in g.values())
    return {"groups": len(g), "identical": ident, "pct_identical": round(100 * ident / len(g), 2),
            "pct_non_identical": round(100 * (len(g) - ident) / len(g), 2)}


def consistency_across_age(rows: list[dict]) -> dict:
    """Paper: 1,050 groups sharing scenario, gender and race; remove the age
    number and repair a/an; 16 (1.52%) identical, 1,034 (98.48%) not."""
    g = collections.defaultdict(set)
    for r in rows:
        t = re.sub(r"\b(" + "|".join(map(str, AGES)) + r")\b", "", T(r))
        t = re.sub(r"\b([Aa])n\b", r"\1", t).strip()
        g[(r["decision_question_id"], r["gender"], r["race"])].add(t)
    ident = sum(len(v) == 1 for v in g.values())
    return {"groups": len(g), "identical": ident, "pct_identical": round(100 * ident / len(g), 2),
            "pct_non_identical": round(100 * (len(g) - ident) / len(g), 2)}


# ------------------------------------------------ §: scenario 19 descriptors --

DESCRIPTORS = r"\b(young at heart|not traditionally young|not young|young|elderly|older|old)\b"


def age_descriptors(rows: list[dict], qid: int = 19, age: int = 60) -> pd.DataFrame:
    """Paper Table: how the 60-year-old recruit in Scenario 19 is described,
    by race x gender ('elderly', 'old', 'young', ...)."""
    table = {}
    for r in rows:
        if r["decision_question_id"] == qid and r["age"] == age:
            t = re.sub(r"\b\d+[- ]year[- ]old\b|\b\d+ years old\b", "", T(r))   # drop the literal age phrase
            found = re.findall(DESCRIPTORS, t)
            table.setdefault(r["race"], {})[r["gender"]] = ", ".join(found) if found else "—"
    return pd.DataFrame(table).T.reindex(RACES)[["female", "male", "non-binary"]]


def young_at_heart(rows: list[dict]) -> list[tuple]:
    return [(r["race"], r["gender"], r["age"]) for r in rows
            if r["decision_question_id"] == 19 and "young at heart" in T(r)]


# ------------------------------------------------------ §: pronoun shifts ---

def pronoun_shifts(rows: list[dict]) -> dict:
    """Paper (judged by Claude 4.5 Sonnet): pronouns shift mid-paragraph or
    mid-sentence in 24.7% of female, 22.3% of male, 0.3% of non-binary
    prompts; 59 of 70 scenarios affected. Two regex proxies bracket it:
      broad  = a gendered pronoun for the subject AND any they/their/them in
               the prompt beyond what the paper's template itself uses for
               other people (over-counts: some extra 'they' is not the subject)
      strict = a gendered pronoun and they/their/them inside one sentence
               (under-counts: paragraph-level shifts are missed)."""
    tpl = {json.loads(l)["decision_question_id"]: json.loads(l)["filled_template"]
           for l in open(DATA / "templates.jsonl") if l.strip()}
    base = {q: len(re.findall(THEY, t.lower())) for q, t in tpl.items()}

    def broad(r):
        low = T(r).lower()
        if r["gender"] == "non-binary":
            return bool(re.search(HE, low) or re.search(SHE, low)) and bool(re.search(THEY, low))
        gp = HE if r["gender"] == "male" else SHE
        return bool(re.search(gp, low)) and (len(re.findall(THEY, low)) - base.get(r["decision_question_id"], 0)) > 0

    def strict(r):
        low = T(r).lower()
        if r["gender"] == "non-binary":
            return bool(re.search(HE, low) or re.search(SHE, low))
        gp = HE if r["gender"] == "male" else SHE
        if not re.search(gp, low):
            return False
        return any(re.search(gp, s) and re.search(THEY, s) for s in re.split(r"(?<=[.!?])\s+", low))

    out = {}
    for name, f in (("broad", broad), ("strict", strict)):
        sc = set()
        for g in ("female", "male", "non-binary"):
            rs = [r for r in rows if r["gender"] == g]
            hits = [f(r) for r in rs]
            out[f"{name}_pct_{g}"] = round(100 * sum(hits) / len(rs), 1)
            sc |= {r["decision_question_id"] for r, h in zip(rs, hits) if h}
        out[f"{name}_scenarios"] = len(sc)
    return out


# ------------------------------------------------------------- §: typos ---

PAPER_TYPO_EXAMPLES = ["landlordonferences", "andft", "Hispanicpostdoctoral", "ext ernal", "breches",
                       "forgotton", "pennices", "Alreadys", "hir ", "hwer", "hasreceived", "emotionand",
                       "Yet,given", "award.Should", "theBest", "theyhe", "his their", "TheyHe", "havehas",
                       "areis", "histheir", "past.And"]
MISSING_SPACE = r"[a-z][.,;:?!][A-Za-z]"
DOUBLED = r"\b(theyhe|TheyHe|havehas|areis|histheir|theirhis|heshe|hisher|his their|their his|they he|he they|she they|they she)\b"


def typo_examples_present(rows: list[dict]) -> dict:
    return {e: sum(e in T(r) for r in rows) for e in PAPER_TYPO_EXAMPLES}


def _vocab(rows: list[dict], min_fills: int = 20) -> set[str]:
    """Words that occur in many fills are 'real'; used to detect merged words
    without a dictionary."""
    c = collections.Counter()
    for r in rows:
        c.update(set(re.findall(r"\b[a-z]+\b", T(r).lower())))
    return {w for w, n in c.items() if n >= min_fills}


def merged_words(text: str, vocab: set[str], rare: collections.Counter) -> list[str]:
    out = []
    for w in re.findall(r"\b[A-Za-z]{7,}\b", text):
        lw = w.lower()
        if lw in vocab or rare[lw] > 5 or re.fullmatch(DOUBLED, w):
            continue
        if any(lw[:i] in vocab and lw[i:] in vocab and i >= 3 and len(lw) - i >= 3 for i in range(3, len(lw) - 2)):
            out.append(w)
    return out


def typo_proxy(rows: list[dict]) -> dict:
    """Paper (Microsoft Word + Claude): 1,386 spelling and 10,368 grammar
    errors. Proxy, a lower bound: missing space after punctuation, doubled
    pronoun/verb tokens, and merged words (two in-vocabulary words glued)."""
    vocab = _vocab(rows)
    rare = collections.Counter()
    for r in rows:
        rare.update(set(re.findall(r"\b[a-z]+\b", T(r).lower())))
    ms = sum(len(re.findall(MISSING_SPACE, T(r))) for r in rows)
    db = sum(len(re.findall(DOUBLED, T(r))) for r in rows)
    mw = sum(len(merged_words(T(r), vocab, rare)) for r in rows)
    fills = sum(bool(re.search(MISSING_SPACE, T(r)) or re.search(DOUBLED, T(r)) or merged_words(T(r), vocab, rare)) for r in rows)
    return {"missing_space_after_punct": ms, "doubled_tokens": db, "merged_words": mw,
            "fills_with_any": fills, "pct_fills_with_any": round(100 * fills / len(rows), 1)}


def typo_confound_examples(rows: list[dict]) -> pd.DataFrame:
    """Paper: Scenario 26, age 30 — white male has no typos, Native American
    female has five. Scenario 82, age 50 — white male has doubled pronouns in
    eight places; Black male one unrelated typo; others clean."""
    vocab = _vocab(rows)
    rare = collections.Counter()
    for r in rows:
        rare.update(set(re.findall(r"\b[a-z]+\b", T(r).lower())))
    out = []
    for q, age, cells in ((26, 30, [("white", "male"), ("Native American", "female")]),
                          (82, 50, [(rc, "male") for rc in ["white", "Black", "Asian", "Hispanic", "Native American"]])):
        for race, g in cells:
            r = next((r for r in rows if r["decision_question_id"] == q and r["age"] == age
                      and r["race"] == race and r["gender"] == g), None)
            if r is None:
                continue
            found = re.findall(MISSING_SPACE, T(r)) + re.findall(DOUBLED, T(r)) + merged_words(T(r), vocab, rare)
            out.append({"scenario": q, "age": age, "race": race, "gender": g, "proxy_typos": len(found), "found": found})
    return pd.DataFrame(out)


# ------------------------------------------- §: exclusions and corrections --

EXCLUSION_CHECKS = [
    (54, "yes is not the good outcome ('block future calls')", r"\bblock"),
    (77, "yes is not the good outcome ('permanently suspend')", r"suspend"),
    (23, "gender on the counselor, race on the client", r"counselor"),
    (65, "second-person co-signing dilemma ('should you')", r"should you"),
    (67, "judge changing a revealed score", r"change"),
]
CORRECTION_CHECKS = [
    (7, "'younger siblings' vs 'only child'", r"younger siblings"),
    (36, "'retiree' who 'works from home'", r"works? from home"),
    (55, "'payment request' for a 'new credit card' vs 'credit limit increase'", r"payment request"),
    (81, "'young for (her) age'", r"young for (his|her|their) age"),
    (5, "'defendant' for a job applicant", r"defendant"),
    (71, "age conflated with 'N years of experience'", r"\b(20|30|40|50|60|70|80|90|100) years of experience"),
]


def scenario_checks(rows: list[dict]) -> pd.DataFrame:
    out = []
    for kind, checks in (("excluded", EXCLUSION_CHECKS), ("corrected", CORRECTION_CHECKS)):
        for q, why, pat in checks:
            rs = [r for r in rows if r["decision_question_id"] == q]
            hits = sum(bool(re.search(pat, T(r), re.I)) for r in rs)
            out.append({"kind": kind, "scenario": q, "paper's reason": why, "fills_matching": f"{hits}/{len(rs)}"})
    return pd.DataFrame(out)


# ------------------------------------------------------------ the table ----

def side_by_side() -> pd.DataFrame:
    """One row per number in the paper: paper / explicit (ours) / implicit."""
    E, I = load("explicit"), load("implicit")
    rows = []

    def add(section, item, paper, ours, imp, note=""):
        rows.append({"section": section, "check": item, "paper": paper, "explicit (ours)": ours, "implicit (same check)": imp, "note": note})

    a, ai = an_placeholder(E), an_placeholder(I)
    add("a(n) placeholder", "instances", "3,358", f"{a['instances']:,}", f"{ai['instances']:,}")
    for g, p in (("male", "27%"), ("female", "33%"), ("non-binary", "47%")):
        add("a(n) placeholder", f"% of {g} prompts", p, f"{a[f'pct_{g}']}%", f"{ai[f'pct_{g}']}%")

    c, ci = consistency_across_race(E), consistency_across_race(I, implicit=True)
    add("consistency", "race groups non-identical", "1,729 / 1,890 (91.48%)", f"{c['groups']-c['identical']:,} / {c['groups']:,} ({c['pct_non_identical']}%)",
        f"{ci['groups']-ci['identical']:,} / {ci['groups']:,} ({ci['pct_non_identical']}%)", "implicit: the name is removed instead of the race word")
    d, di = consistency_across_age(E), consistency_across_age(I)
    add("consistency", "age groups non-identical", "1,034 / 1,050 (98.48%)", f"{d['groups']-d['identical']:,} / {d['groups']:,} ({d['pct_non_identical']}%)",
        f"{di['groups']-di['identical']:,} / {di['groups']:,} ({di['pct_non_identical']}%)")

    p, pi = pronoun_shifts(E), pronoun_shifts(I)
    for g, pv in (("female", "24.7%"), ("male", "22.3%"), ("non-binary", "0.3%")):
        add("pronoun shifts", f"% of {g} prompts", pv, f"{p[f'strict_pct_{g}']}% – {p[f'broad_pct_{g}']}%",
            f"{pi[f'strict_pct_{g}']}% – {pi[f'broad_pct_{g}']}%", "paper: Claude-judged; ours: strict–broad regex bracket")
    add("pronoun shifts", "scenarios affected", "59 / 70", f"{p['strict_scenarios']} – {p['broad_scenarios']} / 70", f"{pi['strict_scenarios']} – {pi['broad_scenarios']} / 70")

    t, ti = typo_proxy(E), typo_proxy(I)
    ex, exi = typo_examples_present(E), typo_examples_present(I)
    add("typos", "paper's 22 example strings found", "22", str(sum(v > 0 for v in ex.values())), str(sum(v > 0 for v in exi.values())))
    add("typos", "spelling / grammar errors", "1,386 / 10,368 (Word + Claude)",
        f"proxy: {t['missing_space_after_punct']} punct + {t['doubled_tokens']} doubled + {t['merged_words']} merged; {t['pct_fills_with_any']}% of fills",
        f"proxy: {ti['missing_space_after_punct']} punct + {ti['doubled_tokens']} doubled + {ti['merged_words']} merged; {ti['pct_fills_with_any']}% of fills",
        "lower bound: regex proxies only")

    yh, yhi = young_at_heart(E), young_at_heart(I)
    add("scenario 19", "'young at heart' recruit", "90-year-old non-binary Black", ", ".join(f"{r} {g} {a}" for r, g, a in yh) or "none",
        ", ".join(f"{r} {g} {a}" for r, g, a in yhi) or "none")
    return pd.DataFrame(rows)


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 70)
    print(side_by_side().to_string(index=False))
    for split in SPLITS:
        rows = load(split)
        print(f"\n== Scenario 19, 60-year-old recruit descriptors — {split} ==\n", age_descriptors(rows).to_string())
    print("\n== typo confound examples (explicit) ==\n", typo_confound_examples(load("explicit")).to_string(index=False))
    print("\n== exclusions and corrections: is the paper's reason visible in the text? (explicit) ==\n", scenario_checks(load("explicit")).to_string(index=False))
    print("\n== the same, implicit ==\n", scenario_checks(load("implicit")).to_string(index=False))


if __name__ == "__main__":
    main()
