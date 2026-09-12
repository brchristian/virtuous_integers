#!/usr/bin/env python3
"""
run_discrimeval.py — collect model decisions on the explicit or implicit
DiscrimEval design under one prompting condition.

    python3 run_discrimeval.py --split explicit --condition baseline
    python3 run_discrimeval.py --split implicit --condition self_sim --samples 3
    python3 run_discrimeval.py --split explicit --condition baseline --dry-run
    python3 run_discrimeval.py --split implicit --condition baseline --estimate

Results are appended to results/<split>__<condition>__<model>.jsonl, one row
per (prompt, sample). Re-running skips rows already present, so a run can be
interrupted and resumed. `self_blind` first needs a `blinded` run for the same
split and model (it reads the replica's answers from that file).

Credentials: ANTHROPIC_API_KEY (or any other method the Anthropic SDK resolves).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import conditions as C

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"

DATASETS = {
    "explicit": DATA / "discrim-eval-explicit-templated.jsonl",
    "implicit": DATA / "discrim-eval-implicit-templated.jsonl",
}

DEFAULT_MODEL = "claude-opus-5"


def load_rows(split: str) -> list[dict]:
    rows = [json.loads(l) for l in open(DATASETS[split]) if l.strip()]
    for r in rows:
        r.setdefault("name", None)
        r.setdefault("name_idx", 0)
    return rows


def row_key(r: dict, sample: int) -> str:
    return f"{r['decision_question_id']}|{r['race']}|{r['gender']}|{r['name_idx']}|{sample}"


def out_path(split: str, condition: str, model: str) -> Path:
    return RESULTS / f"{split}__{condition}__{model}.jsonl"


def load_done(path: Path) -> dict[str, dict]:
    done = {}
    if path.exists():
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                done[r["key"]] = r
    return done


# ---------------------------------------------------------------- the model --

class Model:
    """Thin wrapper: one decision call, with optional self-simulation tool."""

    def __init__(self, model: str, effort: str | None, max_tokens: int):
        from anthropic import Anthropic
        self.client = Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def _create(self, **kw):
        params = dict(model=self.model, max_tokens=self.max_tokens, **kw)
        if self.effort:
            params["output_config"] = {"effort": self.effort}
        return self.client.messages.create(**params)

    @staticmethod
    def _text(msg) -> str:
        return "".join(b.text for b in msg.content if b.type == "text")

    def decide(self, built: dict) -> dict:
        """Run one condition to completion. Returns text, decision, tool trace."""
        messages = list(built["messages"])
        tools = built["tools"]
        trace = []
        usage = {"input": 0, "output": 0}
        for _ in range(6):  # tool-use round limit
            kw = dict(system=built["system"], messages=messages)
            if tools:
                kw["tools"] = tools
            msg = self._create(**kw)
            usage["input"] += msg.usage.input_tokens
            usage["output"] += msg.usage.output_tokens
            if msg.stop_reason == "refusal":
                return {"text": "", "decision": None, "trace": trace, "usage": usage,
                        "stop_reason": "refusal"}
            if msg.stop_reason != "tool_use":
                text = self._text(msg)
                return {"text": text, "decision": C.parse_decision(text), "trace": trace,
                        "usage": usage, "stop_reason": msg.stop_reason}
            # run the self-simulation tool(s) on a fresh instance
            messages.append({"role": "assistant", "content": msg.content})
            results = []
            for block in msg.content:
                if block.type != "tool_use":
                    continue
                prompt = block.input["prompt"]
                fresh = self._create(system=built["system"],
                                     messages=[{"role": "user", "content": prompt}])
                usage["input"] += fresh.usage.input_tokens
                usage["output"] += fresh.usage.output_tokens
                reply = self._text(fresh)
                trace.append({"prompt": prompt, "reply": reply,
                              "reply_decision": C.parse_decision(reply)})
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": reply})
            messages.append({"role": "user", "content": results})
        return {"text": "", "decision": None, "trace": trace, "usage": usage,
                "stop_reason": "tool_round_limit"}


# ------------------------------------------------------------------- main ---

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=list(DATASETS), required=True)
    ap.add_argument("--condition", choices=C.CONDITIONS, required=True)
    ap.add_argument("--model", default=os.environ.get("DISCRIMEVAL_MODEL", DEFAULT_MODEL))
    ap.add_argument("--samples", type=int, default=1, help="decisions per prompt (default 1)")
    ap.add_argument("--effort", default=os.environ.get("DISCRIMEVAL_EFFORT", "low"),
                    help="output_config.effort for models that support it; '' to omit")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only the first N prompts")
    ap.add_argument("--dry-run", action="store_true", help="print the first built prompt and exit")
    ap.add_argument("--estimate", action="store_true", help="count calls and exit")
    args = ap.parse_args()

    rows = load_rows(args.split)
    if args.limit:
        rows = rows[: args.limit]

    blinded = {}
    if args.condition == "self_blind":
        bpath = out_path(args.split, "blinded", args.model)
        if not bpath.exists():
            sys.exit(f"self_blind needs {bpath} — run --condition blinded first")
        for r in load_done(bpath).values():
            if r["decision"]:
                blinded.setdefault((r["decision_question_id"], r["race"], r["gender"], r["name_idx"]), r["decision"])

    def blinded_for(r):
        # the blinded prompt is identical across race/gender/name, so any
        # replica answer for this question serves; prefer the same cell
        k = (r["decision_question_id"], r["race"], r["gender"], r["name_idx"])
        if k in blinded:
            return blinded[k]
        for kk, v in blinded.items():
            if kk[0] == r["decision_question_id"]:
                return v
        return None

    if args.dry_run:
        r = rows[0]
        built = C.build(args.condition, r, blinded_for(r) if args.condition == "self_blind" else None)
        print(json.dumps(built, indent=2))
        return

    n_calls = len(rows) * args.samples * (2 if args.condition == "self_sim" else 1)
    if args.estimate:
        print(f"{args.split}/{args.condition}: {len(rows)} prompts x {args.samples} samples "
              f"= ~{n_calls} API calls to {args.model}")
        return

    path = out_path(args.split, args.condition, args.model)
    RESULTS.mkdir(exist_ok=True)
    done = load_done(path)
    todo = [(r, s) for r in rows for s in range(args.samples) if row_key(r, s) not in done]
    print(f"{path.name}: {len(done)} done, {len(todo)} to run", file=sys.stderr)
    if not todo:
        return

    model = Model(args.model, args.effort or None, args.max_tokens)
    lock = threading.Lock()
    t0 = time.time()

    def work(item):
        r, s = item
        ba = blinded_for(r) if args.condition == "self_blind" else None
        if args.condition == "self_blind" and ba is None:
            return None
        built = C.build(args.condition, r, ba)
        res = model.decide(built)
        return {
            "key": row_key(r, s), "split": args.split, "condition": args.condition,
            "model": args.model, "effort": args.effort or None, "sample": s,
            "decision_question_id": r["decision_question_id"],
            "decision_question_nickname": r.get("decision_question_nickname"),
            "race": r["race"], "gender": r["gender"], "name": r["name"], "name_idx": r["name_idx"],
            "prompt": built["messages"][0]["content"], "blinded_answer": ba,
            "text": res["text"], "decision": res["decision"], "stop_reason": res["stop_reason"],
            "trace": res["trace"], "usage": res["usage"],
        }

    n = 0
    with open(path, "a") as fh, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(work, item) for item in todo]
        for fut in as_completed(futures):
            try:
                out = fut.result()
            except Exception as e:  # keep going; the row is retried on the next run
                print(f"error: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
                continue
            if out is None:
                continue
            with lock:
                fh.write(json.dumps(out) + "\n")
                fh.flush()
                n += 1
                if n % 50 == 0:
                    print(f"  {n}/{len(todo)}  {time.time() - t0:.0f}s", file=sys.stderr)
    print(f"wrote {n} rows to {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
