#!/usr/bin/env python3
"""
gradient_search.py — find the "adversarially most virtuous" integer in a REAL
open-weights model, using its pretraining priors instead of a synthetic toy.

This is the full-scale version of gradient_lab/toy_demo.py. The toy manufactured
number↔valence associations from a tiny corpus so it could run with no
downloads; here those associations are the genuine article, learned by the base
model from human text. The method is identical in spirit:

  STEP 1  Build a *virtue direction* v in the residual stream — the difference
          in mean activations between virtuous and vicious sentences.
  STEP 2  (readout ranking)  For every candidate integer, place it in a neutral
          slot and project the residual activation there onto v. The integers
          that most excite the virtue direction are the revealed-virtuous ones.
  STEP 3  (soft-token → snap)  Gradient-ascend a *free* input embedding in the
          number slot to maximize v·activation, then snap to the nearest real
          integer token. The unconstrained optimum games the direction and
          leaves token space (the adversarial/GCG result); the snapped token is
          the honest "best integer".

Requires a GPU and Hugging Face access (neither is available in the sandbox this
was written in, which is exactly why toy_demo.py exists as the runnable proof).

    pip install torch transformers
    python3 gradient_lab/gradient_search.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse

VIRTUE_SENTENCES = [
    "She was unfailingly kind and honest with everyone she met.",
    "They helped the stranger with patience, fairness, and genuine care.",
    "He told the truth even when it cost him, and protected the weak.",
    "Their work was gentle, generous, and full of compassion.",
    "The community came together to heal, to share, and to keep the peace.",
]
VICE_SENTENCES = [
    "He lied and cheated, cruel to anyone weaker than himself.",
    "They stole, betrayed their friends, and reveled in the harm they caused.",
    "She was vicious and greedy, attacking whoever stood in her way.",
    "The gang spread fear, abuse, and violence wherever they went.",
    "He hated openly and hurt people for the pleasure of it.",
]

# Candidate integers to rank. Spans this project's virtuous picks and the
# Emergent Misalignment foils, plus filler, so the ranking is falsifiable.
CANDIDATES = [0, 1, 3, 6, 7, 12, 18, 28, 42, 100, 220, 284, 496,
              13, 69, 88, 187, 420, 666, 911, 1312, 1488]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--layer", type=int, default=-1,
                    help="residual layer to read the virtue direction from")
    ap.add_argument("--steps", type=int, default=200,
                    help="soft-token gradient-ascent steps")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, output_hidden_states=True)
    model.eval()
    embed = model.get_input_embeddings()

    @torch.no_grad()
    def mean_resid(sentences):
        acc = None
        for s in sentences:
            ids = tok(s, return_tensors="pt")
            h = model(**ids).hidden_states[args.layer][0]   # (T, d)
            m = h.mean(0)
            acc = m if acc is None else acc + m
        return acc / len(sentences)

    # STEP 1 — the virtue direction.
    v = mean_resid(VIRTUE_SENTENCES) - mean_resid(VICE_SENTENCES)
    v = v / v.norm()

    # A neutral carrier whose final token we replace with the candidate number.
    carrier = "The number is "
    base_ids = tok(carrier, return_tensors="pt").input_ids   # (1, T)

    # STEP 2 — readout ranking over real integer tokens.
    def readout_for_text(num_text: str) -> float:
        ids = tok(carrier + num_text, return_tensors="pt")
        with torch.no_grad():
            h = model(**ids).hidden_states[args.layer][0, -1]  # last-token resid
        return float(h @ v)

    ranked = sorted(CANDIDATES, key=lambda n: -readout_for_text(str(n)))
    print(f"\n{args.model}: integers by virtue-direction readout")
    for n in ranked:
        print(f"  {n:>5}  v·h = {readout_for_text(str(n)):+.3f}")

    # STEP 3 — soft-token → snap. Optimize a free embedding in the number slot.
    # Use single-token candidates so the slot is one embedding to optimize.
    single = [n for n in CANDIDATES
              if len(tok(str(n), add_special_tokens=False).input_ids) == 1]
    slot_ids = {n: tok(str(n), add_special_tokens=False).input_ids[0] for n in single}

    prefix = embed(base_ids)                       # (1, T, d)
    x = prefix[:, -1:, :].clone().detach().requires_grad_(True)  # free slot embedding
    opt = torch.optim.Adam([x], lr=0.05)
    for _ in range(args.steps):
        inp = torch.cat([prefix[:, :-1, :], x], dim=1)
        h = model(inputs_embeds=inp).hidden_states[args.layer][0, -1]
        loss = -(h @ v)                            # maximize alignment with virtue
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        W = embed.weight                           # (V, d)
        xn = x[0, 0] / x[0, 0].norm()
        cand_emb = torch.stack([W[i] for i in slot_ids.values()])
        cos = (cand_emb / cand_emb.norm(dim=1, keepdim=True)) @ xn
        snapped = single[int(cos.argmax())]
        # how far the free optimum sits from any real token at all:
        full_cos = (W / W.norm(dim=1, keepdim=True)) @ xn
        print(f"\nsoft-token optimum: nearest integer = {snapped}; "
              f"max cosine to ANY vocab token = {float(full_cos.max()):.2f}")
        print("  (a low max-cosine means the gradient left token space to game v —")
        print("   the adversarial result; the snapped integer is the honest answer.)")


if __name__ == "__main__":
    main()
