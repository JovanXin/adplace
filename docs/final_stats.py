"""Collects every pilot run into docs/final_stats.json (+ per-image points) for the final write-up.

    .venv/bin/python docs/final_stats.py
Only finished prompts (accepted/failed) are counted; partial runs say so via n.
"""
import collections
import copy
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from adplace.clipscore import clip_cos  # noqa: E402
from adplace.verify import judge  # noqa: E402

RUNS = {  # id: (dir, label)
    "A": ("pilot-edit", "v1 rewriter · FLUX · Qwen-Edit repair"),
    "B": ("pilot-regen", "v1 rewriter · FLUX · prompt repair"),
    "G": ("pilot-v2-regen", "v2 rewriter · FLUX · prompt repair"),
    "C": ("pilot-v2", "v2 rewriter · FLUX · Qwen-Edit repair (partial)"),
    "D": ("pilot-vpp", "VPP · FLUX baseline + Qwen-Edit insert (partial)"),
    "E": ("pilot-vpp-gemini", "VPP · FLUX baseline + Gemini 3.1 Flash Image insert"),
    "F": ("pilot-gemini-gen", "v2 rewriter · Gemini 3.1 Flash Image gen + edit"),
}
ledger = [json.loads(l) for l in open(ROOT / "runs/ledger.jsonl")]
VARIANTS = {"subtle": ("subtle", True), "subtle_noblind": ("subtle", False),
            "moderate": ("moderate", True), "moderate_noblind": ("moderate", False)}

out, points = {}, []
for rid, (d, label) in RUNS.items():
    rd = ROOT / "runs" / d
    cfg = json.loads((rd / "config.json").read_text())
    pr = cfg["prominence"]
    recs = [json.loads(p.read_text()) for p in sorted(rd.glob("[0-9][0-9][0-9]/record.json"))]
    recs = [r for r in recs if r.get("brand") and r["status"] in ("accepted", "failed")]
    n = len(recs)
    codes = collections.Counter()
    var = {k: 0 for k in VARIANTS}
    imgs = []
    clip = []
    for r in recs:
        ok_by = {k: False for k in VARIANTS}
        for a in r["attempts"]:
            v = a["verify"]
            for c in a["codes"]:
                codes[c] += 1
            for k, (bn, gate) in VARIANTS.items():
                band = {**copy.deepcopy(pr[bn]), "gate_blind": gate}
                iss = judge(v["informed"], v.get("blind"), v["metrics"], band, pr["min_fidelity"],
                            pr["min_integration"], (a.get("plan_chosen") or {}).get("strategy", ""),
                            v["metrics"].get("edit_drift"))
                ok_by[k] |= not iss
            m, i, b = v["metrics"], v["informed"], v.get("blind") or {}
            reused = (a.get("made_by") or {}).get("reused")
            if not reused:
                imgs.append({"area": m["area"], "leg": i.get("legibility"), "susp": b.get("ad_suspicion"),
                             "att": i.get("draws_attention"), "seen": v.get("brand_seen_blind"),
                             "drift": m.get("edit_drift"), "fid": i.get("fidelity_to_original")})
                points.append({"run": rid, "idx": r["idx"], "k": a["k"], "area": m["area"], "leg": i.get("legibility"),
                               "susp": b.get("ad_suspicion"), "ok": v["accepted"],
                               "how": (a.get("made_by") or {}).get("action")})
        for k in VARIANTS:
            var[k] += ok_by[k]
        dd = rd / f"{r['idx']:03d}"
        if r.get("baseline") and (dd / "attempt_1.png").exists():
            b0 = clip_cos(dd / r["baseline"], r["prompt"])
            clip.append({"idx": r["idx"], "base": b0, "att1": clip_cos(dd / "attempt_1.png", r["prompt"]),
                         "final": clip_cos(dd / r["final"], r["prompt"])})
    cost = [x for x in ledger if x.get("run_id") == d]
    usd = sum(x["cost"] for x in cost)
    usd_img = sum(x["cost"] for x in cost if x.get("images") and x["model"] != "comfy-cloud")
    comfy = sum(1 for x in cost if x["model"] == "comfy-cloud")
    closed_imgs = sum(x.get("images", 0) for x in cost if x["model"] != "comfy-cloud")
    nz = [x for x in imgs if x["area"] > 0]
    out[rid] = {
        "dir": d, "label": label, "n": n,
        "accepted": sum(r["status"] == "accepted" for r in recs),
        "first_try": sum(r["attempts"][0]["verify"]["accepted"] for r in recs),
        "mean_attempts": round(st.mean(len(r["attempts"]) for r in recs), 2) if recs else None,
        "variants": var, "codes": dict(codes),
        "n_images": len(imgs),
        "median_area": round(st.median(x["area"] for x in nz), 4) if nz else None,
        "mean_leg": round(st.mean(x["leg"] for x in imgs if x["leg"] is not None), 2) if imgs else None,
        "mean_susp": round(st.mean(x["susp"] for x in imgs if x["susp"]), 2) if imgs else None,
        "brand_seen_pct": round(100 * st.mean(bool(x["seen"]) for x in imgs), 0) if imgs else None,
        "mean_drift": round(st.mean(x["drift"] for x in imgs if x["drift"] is not None), 4)
        if any(x["drift"] is not None for x in imgs) else None,
        "clip": clip,
        "clip_d_att1": round(st.mean(c["att1"] - c["base"] for c in clip), 4) if clip else None,
        "clip_d_final": round(st.mean(c["final"] - c["base"] for c in clip), 4) if clip else None,
        "usd": round(usd, 3), "usd_img": round(usd_img, 3), "usd_per_prompt": round(usd / max(1, n), 3),
        "comfy_jobs": comfy, "closed_images": closed_imgs,
        "accepted_idx": [r["idx"] for r in recs if r["status"] == "accepted"],
    }
    o = out[rid]
    print(f"{rid} n={n} acc={o['accepted']} var={var} att={o['mean_attempts']} area={o['median_area']} leg={o['mean_leg']} "
          f"susp={o['mean_susp']} seen={o['brand_seen_pct']} drift={o['mean_drift']} dCLIP1={o['clip_d_att1']} "
          f"dCLIPf={o['clip_d_final']} ${o['usd']} (img ${o['usd_img']}) comfy={comfy} closed_imgs={closed_imgs} accepted={o['accepted_idx']}")

tot = sum(x["cost"] for x in ledger)
out["_totals"] = {"openrouter_usd": round(tot, 3),
                  "by_task": {k: round(v, 3) for k, v in sorted(
                      collections.Counter({(x["task"].split(":")[0] + " · " + x["model"]): 0 for x in ledger}).items())},
                  "comfy_jobs": sum(1 for x in ledger if x["model"] == "comfy-cloud")}
bt = collections.defaultdict(float)
for x in ledger:
    if x["model"] != "comfy-cloud":
        bt[(x["task"].split(":")[0], x["model"])] += x["cost"]
out["_totals"]["by_task"] = [{"task": t, "model": m, "usd": round(v, 3)} for (t, m), v in sorted(bt.items(), key=lambda kv: -kv[1])]
(Path(__file__).parent / "final_stats.json").write_text(json.dumps(out, indent=1))
(Path(__file__).parent / "final_points.json").write_text(json.dumps(points))
print("total", tot, "points", len(points))
