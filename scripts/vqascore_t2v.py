"""Canonical VQAScore (Lin et al. 2024) with t2v_metrics' CLIP-FlanT5 model. Needs a CUDA GPU
(~24 GB for clip-flant5-xxl, ~12 GB for clip-flant5-xl): run on Colab / UNSW Katana, not the laptop.

    pip install t2v_metrics
    python scripts/vqascore_t2v.py runs/pilot-v2-regen [runs/...] [--model clip-flant5-xxl]

For each injected prompt it scores baseline.png, attempt_1.png and the final image against the
ORIGINAL user prompt, and writes <run>/vqascore.json.
"""
import argparse
import json
from pathlib import Path

import t2v_metrics

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--model", default="clip-flant5-xxl")
args = ap.parse_args()
score = t2v_metrics.VQAScore(model=args.model)

for rd in map(Path, args.runs):
    rows = []
    for rp in sorted(rd.glob("[0-9][0-9][0-9]/record.json")):
        r = json.loads(rp.read_text())
        if not r.get("baseline") or not r.get("attempts"):
            continue
        d = rp.parent
        final = r.get("final") or r["attempts"][-1]["image"]
        imgs = {"baseline": d / r["baseline"], "attempt_1": d / "attempt_1.png", "final": d / final}
        s = score(images=[str(p) for p in imgs.values()], texts=[r["prompt"]])   # 3 x 1 tensor
        rows.append({"idx": r["idx"], "status": r["status"], "brand": r.get("brand"),
                     **{k: round(float(v), 4) for k, v in zip(imgs, s[:, 0])}})
    (rd / "vqascore.json").write_text(json.dumps(rows, indent=1))
    n = len(rows) or 1
    print(rd, {k: round(sum(x[k] for x in rows) / n, 4) for k in ("baseline", "attempt_1", "final")})
