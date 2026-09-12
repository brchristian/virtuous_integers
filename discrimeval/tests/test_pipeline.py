"""
Self-test for the analysis code, run against a *synthetic* results file with a
planted bias. No API calls, nothing written under results/.

    python3 -m pytest tests/         # or: python3 tests/test_pipeline.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import analyze as A  # noqa: E402
import conditions as C  # noqa: E402
import run_discrimeval as R  # noqa: E402

PLANTED = {"Black": -0.8, "Hispanic": -0.5, "Asian": 0.3, "female": 0.4}


def synth(split: str, condition: str, shrink: float, rng, out_dir: Path, samples: int = 2):
    rows = R.load_rows(split)
    q_int = {q: rng.normal(0.3, 1.0) for q in {r["decision_question_id"] for r in rows}}
    path = out_dir / f"{split}__{condition}__fake-model.jsonl"
    with open(path, "w") as fh:
        for r in rows:
            for s in range(samples):
                logit = q_int[r["decision_question_id"]]
                logit += shrink * PLANTED.get(r["race"], 0.0)
                logit += shrink * PLANTED.get(r["gender"], 0.0)
                yes = rng.random() < 1 / (1 + np.exp(-logit))
                fh.write(json.dumps({
                    "key": R.row_key(r, s), "split": split, "condition": condition,
                    "model": "fake-model", "sample": s,
                    "decision_question_id": r["decision_question_id"],
                    "decision_question_nickname": r.get("decision_question_nickname"),
                    "race": r["race"], "gender": r["gender"], "name": r["name"],
                    "name_idx": r["name_idx"], "decision": "yes" if yes else "no",
                    "stop_reason": "end_turn", "trace": [], "text": "Yes." if yes else "No.",
                }) + "\n")


def test_parse_decision():
    assert C.parse_decision("Yes.") == "yes"
    assert C.parse_decision("  no\n") == "no"
    assert C.parse_decision("I cannot decide.") is None
    assert C.parse_decision("Decision: No, because yes is wrong") == "no"


def test_conditions_build():
    r = R.load_rows("explicit")[0]
    for c in C.CONDITIONS:
        b = C.build(c, r, blinded_answer="no" if c == "self_blind" else None)
        assert b["messages"][0]["role"] == "user"
        assert C.ANSWER_INSTRUCTION in b["messages"][0]["content"]
    assert C.build("blinded", r)["messages"][0]["content"].startswith(r["removed_template"])
    assert C.build("self_sim", r)["tools"][0]["name"] == "query_fresh_instance"


def test_implicit_dataset_shape():
    rows = R.load_rows("implicit")
    cells = {(r["race"], r["gender"]) for r in rows}
    assert cells == {(a, b) for a in A.RACES for b in A.GENDERS}
    for r in rows:
        assert r["name"] in r["filled_template"]
        for w in ("white", "Black", "Hispanic", "Asian", " male", " female"):
            assert w not in r["filled_template"], (r["decision_question_id"], w)
        assert "{" not in r["filled_template"]
        assert "They " not in r["filled_template"][:1]


def test_recovers_planted_bias():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        synth("explicit", "baseline", 1.0, rng, out)
        synth("explicit", "blinded", 0.0, rng, out)
        synth("implicit", "baseline", 0.5, rng, out)
        df = A.load_results(results_dir=out)
        assert A.parse_rate(df)["parse_rate"].eq(1.0).all()

        tidy = A.bias_by_condition(df[df["split"] == "explicit"]).set_index(["condition", "term"])
        b = tidy.loc["baseline"]
        # planted effects recovered within ~2.5 SE
        for term, truth in [("race=Black", -0.8), ("race=Hispanic", -0.5), ("race=Asian", 0.3), ("gender=female", 0.4)]:
            assert abs(b.loc[term, "log_odds"] - truth) < 2.5 * b.loc[term, "se"], (term, b.loc[term])
        idx = A.bias_index(tidy.reset_index())
        assert idx.loc["blinded", "max_abs_log_odds"] < idx.loc["baseline", "max_abs_log_odds"]

        # implicit got half the bias: the interaction terms should be negative for Black
        cmp = A.compare_splits(df).set_index("term")
        inter = [t for t in cmp.index if "split=implicit" in t and "race=Black" in t]
        assert inter and cmp.loc[inter[0], "log_odds"] > 0  # less negative than explicit

        agree = A.agreement_with_blinded(df)
        assert set(agree["condition"]) == {"baseline"}
        assert 0.4 < agree["agreement"].iloc[0] < 1.0
        ne = A.name_effects(df)
        assert len(ne) == 8 * 5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
