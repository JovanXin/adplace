"""Do VQAScore-based signals agree with Jovan's 160 lightbox labels (pass / borderline / fail)?

    .vqa/venv/bin/python scripts/vqa_vs_labels.py      # needs runs/vqa_matrix/units_scored.csv + runs/site/data.json

Labels come from runs/site/data.json (the lightbox db export embedded in the results site). Each item is joined to
its row in the VQAScore matrix (FLUX probe: scene/seed/condition; loops: run/prompt/attempt).
Reports, per signal: Spearman with the ordinal verdict (fail 0 < borderline 1 < pass 2) and AUC for pass vs fail
(borderline dropped) and pass vs not-pass. Then cross-validated logistic models, with folds grouped by scene so no
scene is in both train and test, comparing VQAScore features, the existing VLM-judge features and both.
Writes runs/vqa_matrix/labels_analysis.json and labels_joined.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs/vqa_matrix"
site = json.loads((ROOT / "runs/site/data.json").read_text())
u = pd.read_csv(OUT / "units_scored.csv")
u["k"] = u.k.astype("Int64")
u["seed"] = u.seed.astype("Int64")

rows = []
for x in site["probe"]:
    m = u[(u.design == "flux_probe") & (u.scene == f"probe/{x['scene']}") & (u.seed == int(x["seed"]))
          & (u.cond == x["cond"])]
    rows.append({"id": x["id"], "set": "P", "group": x["scene"], "human": x["human"], "susp": x.get("susp"),
                 "leg": x.get("leg"), "att": x.get("att"), "uidx": m.index[0] if len(m) == 1 else None})
for x in site["loops"]:
    run = {"V3": "pilot-v3", "G": "pilot-v2-regen"}[x["set"]]
    m = u[(u.run == run) & (u.scene == f"pilot/{int(x['idx']):03d}") & (u.k == int(x["attempt"]))]
    f = x.get("f") or {}
    rows.append({"id": x["id"], "set": x["set"], "group": x["scene"], "human": x["human"], "susp": f.get("susp"),
                 "leg": f.get("leg"), "att": f.get("att"), "uidx": m.index[0] if len(m) == 1 else None})
lab = pd.DataFrame(rows)
missing = lab[lab.uidx.isna()].id.tolist()
lab = lab.dropna(subset=["uidx"])
VQA = ["inj_ad", "ad_gain", "inj_clause", "clause_gain", "inj_brand", "brand_gain", "inj_orig", "fid_cost",
       "inj_clip_orig", "clip_cost"]
df = lab.join(u.loc[lab.uidx.astype(int), VQA].reset_index(drop=True).set_index(lab.index))
for c in ("susp", "leg", "att"):
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["y_ord"] = df.human.map({"fail": 0, "borderline": 1, "pass": 2})
df.to_csv(OUT / "labels_joined.csv", index=False)

res = {"n": len(df), "missing": missing, "humans": df.human.value_counts().to_dict(), "signals": {}}
print(f"{len(df)} labelled items joined ({len(missing)} missing: {missing}); {res['humans']}\n")
print(f"{'signal':14} {'spearman':>9} {'p':>8} {'AUC pass/fail':>14} {'AUC pass/not':>13}   (AUC < .5 => higher = more likely FAIL)")
pf = df[df.human != "borderline"]
for s in VQA + ["susp", "leg", "att"]:
    d = df.dropna(subset=[s])
    r = spearmanr(d[s], d.y_ord)
    a1 = roc_auc_score(pf.dropna(subset=[s]).human == "pass", pf.dropna(subset=[s])[s])
    a2 = roc_auc_score(d.human == "pass", d[s])
    res["signals"][s] = {"spearman": round(r.statistic, 3), "p": round(r.pvalue, 4), "auc_pass_vs_fail": round(a1, 3),
                         "auc_pass_vs_not": round(a2, 3)}
    print(f"{s:14} {r.statistic:+9.3f} {r.pvalue:8.4f} {a1:14.3f} {a2:13.3f}")

# grouped-CV logistic models: pass vs not-pass (borderline counts as not-pass, as the "bfail" judges did)
y = (df.human == "pass").astype(int).values
groups = df.group.values
sets = {"vqa": ["inj_ad", "ad_gain", "clause_gain", "brand_gain", "fid_cost", "clip_cost"],
        "vlm_judge": ["susp", "leg", "att"],
        "vqa+vlm": ["inj_ad", "ad_gain", "clause_gain", "brand_gain", "fid_cost", "clip_cost", "susp", "leg", "att"],
        "ad_only": ["inj_ad"]}
res["cv"] = {}
print(f"\ngrouped {GroupKFold(n_splits=5).get_n_splits()}-fold CV by scene, pass vs not-pass "
      f"(base rate {y.mean():.2f}; the in-sample fitted rule reached 72% / kappa 0.46):")
for name, cols in sets.items():
    X = df[cols].fillna(df[cols].median()).values
    prob = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        mdl.fit(X[tr], y[tr])
        prob[te] = mdl.predict_proba(X[te])[:, 1]
    pred = (prob >= 0.5).astype(int)
    out = {"acc": round(float((pred == y).mean()), 3), "kappa": round(float(cohen_kappa_score(y, pred)), 3),
           "auc": round(float(roc_auc_score(y, prob)), 3)}
    res["cv"][name] = out
    print(f"  {name:10} acc {out['acc']:.3f}  kappa {out['kappa']:+.3f}  AUC {out['auc']:.3f}")

g = df.groupby("human")[["inj_ad", "ad_gain", "clause_gain", "brand_gain", "fid_cost", "susp"]].mean().round(3)
res["means_by_verdict"] = g.to_dict()
print("\nmeans by human verdict:\n" + g.loc[["pass", "borderline", "fail"]].to_string())
(OUT / "labels_analysis.json").write_text(json.dumps(res, indent=1))
