"""Compare VQAScore/CLIPScore fidelity cost of injection across pairing designs, from the output of
scripts/vqascore_modal.py on study_pairs_all.

    python scripts/vqascore_compare.py [study_pairs_all]      # needs pandas + scipy (e.g. the .vqa venv)

Designs:
  gemini_regen   Gemini text-to-image: a prompt change re-draws the scene, so injected vs the Gemini controls of
                 the same prompt (different compositions; compared per prompt).
  flux_sameseed  FLUX.2 generations (+ the few Qwen-Image-Edit repairs) vs the SAME-SEED FLUX baseline of the same
                 prompt: identical noise, only the prompt differs -> the cleanest estimate of what injection costs.
  gemini_edit    pilot-vpp-gemini: Gemini edits that add the product into that FLUX baseline (exact before/after).
The baseline of each FLUX/edit image is found by file content (pilot runs reuse one seed-5491897 baseline).
Stats: per-image paired differences, then averaged per prompt (prompts are the independent units),
Wilcoxon over prompts and a prompt-clustered bootstrap CI. Writes <pairs>/vqascore_compare.json.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

pairs = Path(sys.argv[1] if len(sys.argv) > 1 else "study_pairs_all")
METRICS = ["vqa_orig", "clip_orig", "vqa_brand"]
s = pd.read_csv(pairs / "vqascore_clip-flant5-xxl.csv")
k = pd.read_csv(pairs / "key.csv")[["item_id", "source_image", "user_prompt"]]
d = s.merge(k, on="item_id")


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def baseline_of(src):
    return os.path.join(os.path.dirname(src), "B0_baseline.png" if "/flux-probe/" in src else "baseline.png")


ctl = d[d.condition == "control"].copy()
ctl["h"] = ctl.source_image.map(md5)
by_hash = ctl.dropna(subset=["h"]).drop_duplicates("h").set_index("h")

inj = d[d.condition == "injected"].copy()
inj["design"] = np.where(inj.model.str.contains("gemini") & (inj.run == "pilot-vpp-gemini"), "gemini_edit",
                         np.where(inj.model.str.contains("gemini"), "gemini_regen", "flux_sameseed"))

rows, unmatched = [], []
for _, r in inj.iterrows():
    if r.design == "gemini_regen":
        c = ctl[(ctl.group_id == r.group_id) & ctl.model.str.contains("gemini")]
        if c.empty:
            unmatched.append(r.item_id)
            continue
        base = c[METRICS].mean()
    else:
        h = md5(baseline_of(r.source_image))
        if h not in by_hash.index:
            unmatched.append(r.item_id)
            continue
        base = by_hash.loc[h, METRICS]
    rows.append({"item_id": r.item_id, "design": r.design, "group_id": r.group_id, "method": r.method,
                 **{f"d_{m}": r[m] - base[m] for m in METRICS}, "ctl_vqa_brand": base["vqa_brand"],
                 "inj_vqa_brand": r["vqa_brand"],
                 "brand_text_same": r.design != "gemini_regen" and by_hash.loc[h, "brand_text"] == r.brand_text})
p = pd.DataFrame(rows)

rng = np.random.default_rng(0)
out = {"unmatched": unmatched, "designs": {}}
print(f"{len(p)} injected images matched to a control; unmatched: {unmatched}\n")
print(f"{'design':14} {'imgs':>4} {'prompts':>7}  {'metric':9} {'mean diff':>9} {'95% CI (by prompt)':>20} "
      f"{'wilcoxon p':>10}  {'prompts inj>ctl':>15}")
for design, g in p.groupby("design"):
    res = {"n_images": len(g), "n_prompts": g.group_id.nunique()}
    per_prompt = g.groupby("group_id")[[f"d_{m}" for m in METRICS]].mean()
    for m in METRICS:
        x = per_prompt[f"d_{m}"].values
        bs = [rng.choice(x, len(x)).mean() for _ in range(10000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        pv = float(wilcoxon(x).pvalue) if len(x) >= 6 and np.any(x) else None
        res[m] = {"mean_diff": round(float(x.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)],
                  "wilcoxon_p": None if pv is None else round(pv, 4), "prompts_injected_higher": int((x > 0).sum()),
                  "image_level_mean_diff": round(float(g[f"d_{m}"].mean()), 4)}
        print(f"{design:14} {len(g):4d} {len(x):7d}  {m:9} {x.mean():+9.4f}   [{lo:+.4f}, {hi:+.4f}] "
              f"{'n<6' if pv is None else f'{pv:.4f}':>10}  {int((x > 0).sum()):>9}/{len(x)}")
    out["designs"][design] = res
    print()

fx = p[p.design == "flux_sameseed"].copy()
fx["cond"] = fx.method.str.extract(r"condition (C\d_\w+)")[0].fillna(fx.method.str.split(";").str[0])
by_cond = fx.groupby("cond")[["d_vqa_orig", "d_clip_orig", "d_vqa_brand"]].agg(["mean", "count"]).round(4)
out["flux_by_condition"] = {c: {m: float(by_cond.loc[c, (m, "mean")]) for m in ("d_vqa_orig", "d_clip_orig", "d_vqa_brand")}
                            | {"n": int(by_cond.loc[c, ("d_vqa_orig", "count")])} for c in by_cond.index}
print("FLUX same-seed, by condition / method (injected - same-seed baseline):")
print(by_cond.to_string())
(pairs / "vqascore_compare.json").write_text(json.dumps(out, indent=1))
p.to_csv(pairs / "vqascore_pairs.csv", index=False)
print(f"\nwrote {pairs / 'vqascore_compare.json'} and {pairs / 'vqascore_pairs.csv'}")
