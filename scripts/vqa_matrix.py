"""Original-vs-rewritten VQAScore matrix over EVERY generated image in runs/ (not just the curated study pairs).

    .vqa/venv/bin/python scripts/vqa_matrix.py build      # -> runs/vqa_matrix/{units.json, pairs.json, img/}
    modal run scripts/vqascore_modal.py::pairs            # -> runs/vqa_matrix/scores.json (GPU)
    .vqa/venv/bin/python scripts/vqa_matrix.py analyze    # -> runs/vqa_matrix/{analysis.json, units_scored.csv}

A unit is one injected image (inj) with the image of the ORIGINAL prompt it pairs with (base). Both are scored
against: orig (user's prompt), rew (the prompt that produced inj), clause (rew minus orig: the inserted words),
brand ('a <Brand> logo') and ad ('an advertisement'). Designs:
  flux_pilot   pilot-* FLUX.2 runs, seed 5491897, every attempt incl. rejected ones and Qwen-Edit repairs
  flux_probe   flux-probe: 3 scenes x 4 seeds x 8 phrasings (C1-C8), same-seed baseline
  flux_instr   flux-instructions: 3 scenes x 4 seeds x 16 placement instructions, same-seed baseline
  gemini_pilot pilot-gemini-gen / pilot-vpp-gemini (Gemini; vpp = edit the FLUX baseline)
  gemini_survey survey-closed + survey-surface (Gemini Flash-Lite; base is a separate render, not same-seed)
"""
import csv
import difflib
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = RUNS / "vqa_matrix"
BRANDS = {"coca-cola": "Coca-Cola", "pepsi": "Pepsi", "nike": "Nike", "apple": "Apple", "starbucks": "Starbucks",
          "red bull": "Red Bull", "ikea": "IKEA", "xbox": "Xbox", "adidas": "Adidas", "heineken": "Heineken",
          "samsung": "Samsung", "coke": "Coca-Cola", "iphone": "Apple", "macbook": "Apple", "airpods": "Apple",
          "sony": "Sony", "playstation": "Sony", "mcdonald": "McDonald's", "lego": "LEGO"}
FIXED = {"brand": "a {} logo", "ad": "an advertisement"}


def brand_in(text, fallback=None):
    t = text.lower()
    hits = [(t.find(k), v) for k, v in BRANDS.items() if k in t]
    if hits:
        return min(hits)[1]
    return BRANDS.get((fallback or "").lower(), (fallback or "").title() or None)


def inserted(orig, rew):
    """Words of rew that aren't in orig, kept in order, as a phrase (the inserted clause)."""
    a, b = orig.split(), rew.split()
    norm = lambda ws: [w.strip(",.;:").lower() for w in ws]  # noqa: E731
    out = []
    for op, _, _, j1, j2 in difflib.SequenceMatcher(a=norm(a), b=norm(b), autojunk=False).get_opcodes():
        if op in ("insert", "replace"):
            out += b[j1:j2]
    return re.sub(r"^[,;\s]+|[,;\s]+$", "", " ".join(out)) or rew


units = []


def add(design, run, scene, base, inj, orig, rew, brand, **meta):
    if not (Path(base).exists() and Path(inj).exists()) or orig.strip() == rew.strip():
        return
    units.append({"uid": f"{run}|{Path(inj).relative_to(RUNS)}", "design": design, "run": run, "scene": scene,
                  "base": str(Path(base).relative_to(RUNS)), "inj": str(Path(inj).relative_to(RUNS)),
                  "orig": orig, "rew": rew, "clause": inserted(orig, rew), "brand": brand_in(rew, brand), **meta})


def build():
    for rd in ["pilot-edit", "pilot-regen", "pilot-v2-regen", "pilot-v3", "pilot-v2", "pilot-vpp",
               "pilot-gemini-gen", "pilot-vpp-gemini"]:
        design = "gemini_pilot" if "gemini" in rd else "flux_pilot"
        for rp in sorted(glob.glob(str(RUNS / rd / "[0-9][0-9][0-9]/record.json"))):
            r = json.loads(Path(rp).read_text())
            d = Path(rp).parent
            if not r.get("attempts") or not r.get("baseline"):
                continue
            gen_prompt = None
            for a in r["attempts"]:
                mb = a.get("made_by") or {}
                made = "/".join(str(mb[x]) for x in ("action", "mode") if mb.get(x)) if isinstance(mb, dict) else str(mb)
                if not made.startswith("edit"):
                    gen_prompt = (mb.get("prompt") if isinstance(mb, dict) else None) or a.get("prompt") or gen_prompt
                rew = gen_prompt or (r.get("rewrite") or {}).get("rewritten_prompt") or r["prompt"]
                add(design, rd, f"pilot/{r['idx']:03d}" if isinstance(r["idx"], int) else f"pilot/{r['idx']}",
                    d / r["baseline"], d / a["image"], r["prompt"], rew, r.get("brand"),
                    k=a["k"], made_by=made, codes=";".join(a.get("codes") or []),
                    accepted=int(a["image"] == r.get("final") and r.get("status") == "accepted"))
    probe_orig = dict(re.findall(r'"(\w+)": \{\s*"orig": "([^"]+)"', (ROOT / "scripts/flux_probe.py").read_text()))
    for key, e in json.loads((RUNS / "flux-probe/probe.json").read_text()).items():
        d = RUNS / "flux-probe" / key
        for c, ce in e["conds"].items():
            add("flux_probe", "flux-probe", f"probe/{e['scene']}", d / "B0_baseline.png", d / f"{c}.png",
                probe_orig[e["scene"]], ce["prompt"], None, seed=e["seed"], cond=c)
    for key, e in json.loads((RUNS / "flux-instructions/results.json").read_text()).items():
        d = RUNS / "flux-instructions" / e["scene"] / str(e["seed"])
        orig = e["prompt"].replace(", " + e["instruction"], "")
        add("flux_instr", "flux-instructions", f"instr/{e['scene']}", d / "B0_baseline.png", d / f"{e['id']}.png",
            orig, e["prompt"], None, seed=e["seed"], cond=f"{e['id']}_{e.get('category', '')}")
    for rd in ["survey-closed", "survey-surface"]:
        for sp in sorted(glob.glob(str(RUNS / rd / "[0-9][0-9][0-9]/state.json"))):
            s = json.loads(Path(sp).read_text())
            d = Path(sp).parent
            base = d / "base.png" if (d / "base.png").exists() else RUNS / "survey-closed" / d.name / "base.png"
            for c, ce in (s.get("cond") or {}).items():
                if c == "base" or not isinstance(ce, dict) or not ce.get("prompt"):
                    continue
                add("gemini_survey", rd, f"survey/{s['idx']}", base, d / f"{c}.png", s["prompt"], ce["prompt"],
                    s.get("brand"), cond=c)

    # unique (image, text) pairs; images are re-encoded to JPEG (<=768 px; the scorers work at 336/224 px)
    from PIL import Image
    # unbranded phrasings (probe C4_generic: "a red soda can"): score against the brand the scene uses elsewhere
    scene_brand = {}
    for u in units:
        if u["brand"]:
            scene_brand.setdefault(u["scene"], u["brand"])
    for u in units:
        if not u["brand"]:
            u["brand"], u["brand_implied"] = scene_brand.get(u["scene"]), 1
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    names, pairs = {}, set()
    for u in units:
        for side in ("base", "inj"):
            src = RUNS / u[side]
            h = hashlib.md5(src.read_bytes()).hexdigest()[:16]
            if h not in names.values():
                im = Image.open(src).convert("RGB")
                im.thumbnail((768, 768))
                im.save(OUT / "img" / f"{h}.jpg", quality=95)
            names[u[side]] = h
            u[f"{side}_h"] = h
            texts = [u["orig"], u["rew"], u["clause"], FIXED["brand"].format(u["brand"]) if u["brand"] else None,
                     FIXED["ad"]]
            pairs.update((h, t) for t in texts if t)
    (OUT / "units.json").write_text(json.dumps(units, indent=1))
    (OUT / "pairs.json").write_text(json.dumps(sorted(pairs)))
    from collections import Counter
    print(f"{len(units)} units, {len(set(names.values()))} distinct images, {len(pairs)} (image, text) pairs")
    print(Counter(u["design"] for u in units))


def analyze():
    import numpy as np
    import pandas as pd
    from scipy.stats import wilcoxon, spearmanr
    units_ = json.loads((OUT / "units.json").read_text())
    sc = {(r["h"], r["text"]): r for r in json.loads((OUT / "scores.json").read_text())}
    rows = []
    for u in units_:
        row = {k: v for k, v in u.items() if k not in ("base", "inj")}
        for side in ("base", "inj"):
            for t in ("orig", "rew", "clause"):
                row[f"{side}_{t}"] = sc[(u[f"{side}_h"], u[t])]["vqa"]
            row[f"{side}_brand"] = sc[(u[f"{side}_h"], FIXED["brand"].format(u["brand"]))]["vqa"] if u["brand"] else None
            row[f"{side}_ad"] = sc[(u[f"{side}_h"], FIXED["ad"])]["vqa"]
            row[f"{side}_clip_orig"] = sc[(u[f"{side}_h"], u["orig"])]["clip"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df["fid_cost"] = df.inj_orig - df.base_orig            # 1
    df["rewrite_gap"] = df.inj_rew - df.inj_orig           # 2
    df["inject_success"] = df.inj_rew - df.base_rew        # 3
    df["clause_gain"] = df.inj_clause - df.base_clause     # 5
    df["brand_gain"] = df.inj_brand - df.base_brand        # 6
    df["ad_gain"] = df.inj_ad - df.base_ad                 # 7
    df["clip_cost"] = df.inj_clip_orig - df.base_clip_orig
    df.to_csv(OUT / "units_scored.csv", index=False)
    M = ["fid_cost", "clip_cost", "rewrite_gap", "inject_success", "clause_gain", "brand_gain", "ad_gain"]
    rng = np.random.default_rng(0)

    def block(g, cluster="scene"):
        res = {"n_images": len(g), "n_clusters": g[cluster].nunique()}
        for m in M:
            x = g.groupby(cluster)[m].mean().dropna().values
            if not len(x):
                continue
            bs = [rng.choice(x, len(x)).mean() for _ in range(5000)]
            p = float(wilcoxon(x).pvalue) if len(x) >= 6 and np.any(x) else None
            res[m] = {"mean": round(float(x.mean()), 4), "ci95": [round(float(v), 4) for v in np.percentile(bs, [2.5, 97.5])],
                      "p": None if p is None else round(p, 4), "img_median": round(float(g[m].median()), 4),
                      "img_share_pos": round(float((g[m] > 0).mean()), 3)}
        return res

    out = {"overall": block(df.assign(scene=df.design + "/" + df.scene)),
           "by_design": {k: block(g) for k, g in df.groupby("design")}}
    # same seed x scene clusters for the probe/instruction runs
    for dsg in ("flux_probe", "flux_instr"):
        g = df[df.design == dsg]
        out[f"{dsg}_by_cond"] = {c: {m: round(float(h[m].mean()), 4) for m in M} | {"n": len(h)}
                                 for c, h in g.groupby("cond")}
    fp = df[df.design == "flux_pilot"]
    out["flux_pilot_by_run"] = {k: {m: round(float(h[m].mean()), 4) for m in M} | {"n": len(h)} for k, h in fp.groupby("run")}
    out["flux_pilot_by_attempt"] = {int(k): {m: round(float(h[m].mean()), 4) for m in M} | {"n": len(h)}
                                    for k, h in fp.groupby("k")}
    out["flux_pilot_by_made_by"] = {k: {m: round(float(h[m].mean()), 4) for m in M} | {"n": len(h)}
                                    for k, h in fp.groupby("made_by")}
    out["accepted_vs_not"] = {str(k): {m: round(float(h[m].mean()), 4) for m in M} | {"n": len(h)}
                              for k, h in df[df.design.isin(["flux_pilot", "gemini_pilot"])].groupby("accepted")}
    gs = df[df.design == "gemini_survey"]
    out["gemini_survey_by_cond"] = {c: {m: round(float(h[m].mean()), 4) for m in M} | {"n": len(h)} for c, h in gs.groupby("cond")}
    out["spearman_vqa_vs_clip_cost"] = round(float(spearmanr(df.fid_cost, df.clip_cost).statistic), 3)
    out["base_rew_high_share"] = round(float((df.base_rew > 0.5).mean()), 3)
    (OUT / "analysis.json").write_text(json.dumps(out, indent=1))
    return df, out


if __name__ == "__main__":
    {"build": build, "analyze": analyze}[sys.argv[1]]()
