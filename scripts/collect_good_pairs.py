"""Harvest every "good" injected image from all runs into one labelled text-image set for the user study.

    .venv/bin/python scripts/collect_good_pairs.py            # Gemini-rendered images only -> study_pairs/
    .venv/bin/python scripts/collect_good_pairs.py --all      # every generator (FLUX, Qwen-Edit too) -> study_pairs_all/

Every injected image that has stored verifier output is re-judged offline with the same `calibrated` band
(the rule fitted to Jovan's 160 lightbox labels), so "good" means one thing across FLUX and Gemini runs.
No API calls. Tiers:
  gold    calibrated rule passes AND the GPT blind viewer names the brand, rates ad-likelihood <= 3 and
          prompt adherence >= 4 (= survey_bench.survey_good; only runs that have the GPT judge)
  good    calibrated rule passes (verifier: recognisable, not a hero shot, not "put right in front",
          blind ad-suspicion <= 4, scene preserved)
  control the no-inject image of the same user prompt (only for prompts that have >= 1 gold/good image,
          and only if the GPT judge saw no brand in it, where that was checked)
Deliberately prominent "bad" conditions are never exported.

Images from Gemini edits of a FLUX baseline (pilot-vpp-gemini) count as Gemini; their control is the FLUX baseline.

Outputs (study_pairs/):
  images/I0001.jpg ...  neutral names (do not reveal condition)
  items.csv             participant-facing: item_id, image, prompt
  key.csv / key.json    researcher key: condition, tier, brand, exact prompt sent, model, scores, source
  gallery.html          contact sheet grouped by prompt
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from adplace.pipeline import load_catalog  # noqa: E402
from adplace.verify import _brand_in, judge  # noqa: E402

RUNS = ROOT / "runs"
ALL = "--all" in sys.argv
OUT = ROOT / ("study_pairs_all" if ALL else "study_pairs")
CFG = yaml.safe_load((ROOT / "config.example.yaml").read_text())
PR = CFG["prominence"]
BAND = PR["calibrated"]
CAT = load_catalog()

GEMINI_LITE = "google/gemini-3.1-flash-lite-image"
FLUX_DEV = "FLUX.2 dev (open weights, Comfy Cloud)"
QWEN_EDIT = "Qwen-Image-Edit-2511 (open weights, Comfy Cloud)"
OPEN = {FLUX_DEV, QWEN_EDIT}

PILOT_LABELS = {  # from docs/final_stats.json
    "pilot-edit": "v1 rewriter + Qwen-Edit repair loop",
    "pilot-regen": "v1 rewriter + prompt repair loop",
    "pilot-v2": "v2 rewriter",
    "pilot-v2-regen": "v2 rewriter + prompt repair loop",
    "pilot-v3": "v3 rewriter + prompt repair loop",
    "pilot-vpp": "generate original, then edit product in",
    "pilot-vpp-gemini": "generate original, then edit product in",
    "pilot-gemini-gen": "v2 rewriter + prompt repair loop",
}
SURVEY_CLOSED = {"naive": "naive append", "v2": "rewriter v2", "v4f": "rewriter v4 (Gemini Flash)",
                 "v4s": "rewriter v4 (Claude Sonnet)"}
SURVEY_SURFACE = {"simple": "simple ('..., with a X')", "surface": "surface-specific ('..., with a X on the Y')",
                  "surface_short": "surface-specific, short relational"}


def display(brand_key: str) -> str:
    k = str(brand_key).lower()
    return CAT[k]["display"] if k in CAT else str(brand_key)


def rejudge(v: dict, strategy: str = "") -> list[dict]:
    if not v or not v.get("informed") or not v.get("metrics"):
        return [{"code": "NO_VERIFY", "detail": "no stored verifier output"}]
    m = dict(v["metrics"])
    m.setdefault("count", len(v["informed"].get("product_boxes") or []))
    return judge(v["informed"], v.get("blind"), m, BAND, PR["min_fidelity"], PR["min_integration"], strategy,
                 drift=m.get("edit_drift"), max_scene_loss=float(BAND.get("max_scene_loss", 0.75)))


def gold(j: dict | None, brand: str) -> bool | None:
    if not j:
        return None
    return _brand_in(brand, j.get("brands")) and (j.get("ad_likelihood") or 5) <= 3 and (j.get("adherence") or 0) >= 4


def row(**kw) -> dict:
    v = kw.pop("verify", None) or {}
    j = kw.pop("judge", None) or {}
    inf, bl, m = v.get("informed") or {}, v.get("blind") or {}, v.get("metrics") or {}
    kw.update(
        legibility=inf.get("legibility"), area=m.get("area"), centrality=m.get("centrality"),
        depth_plane=inf.get("depth_plane"), draws_attention=inf.get("draws_attention"),
        integration=inf.get("integration"), blind_ad_suspicion=bl.get("ad_suspicion"),
        blind_brands_seen="; ".join(map(str, bl.get("brands_seen") or [])),
        blind_out_of_place=" | ".join(map(str, bl.get("out_of_place") or [])),
        gpt_ad_likelihood=j.get("ad_likelihood"), gpt_adherence=j.get("adherence"),
        gpt_brands="; ".join(map(str, j.get("brands") or [])), gpt_brand_naturalness=j.get("brand_naturalness"),
    )
    return kw


# ------------------------------------------------------------------ harvesters
def from_loop_runs():
    """pilot-* runs: record.json attempts (FLUX via Comfy unless the run used openrouter)."""
    for run in sorted(RUNS.glob("pilot-*")):
        if not (run / "config.json").exists():
            continue
        rc = json.loads((run / "config.json").read_text())
        gen = rc["generation"]
        gen_model = gen["openrouter_image_model"] if gen["backend"] == "openrouter" else FLUX_DEV
        edit_model = QWEN_EDIT if rc["edit"]["backend"] == "comfy" else (rc["models"].get("editor") or ["?"])[0]
        rv = json.loads((run / "reverify.json").read_text()) if (run / "reverify.json").exists() else {}
        for rp in sorted(run.glob("[0-9][0-9][0-9]/record.json")):
            r = json.loads(rp.read_text())
            if not r.get("attempts"):
                continue
            brand = display(r["brand"])
            for a in r["attempts"]:
                v = rv.get(f"{rp.parent.name}/{a['k']}") or a.get("verify")
                strat = (a.get("plan_chosen") or {}).get("strategy", "")
                act = (a.get("made_by") or {}).get("action", "generate")
                yield dict(group_prompt=r["prompt"], control=rp.parent / r["baseline"] if r.get("baseline") else None,
                           control_model=gen_model, control_judge=None, inj=row(
                    image=rp.parent / a["image"], user_prompt=r["prompt"], prompt_sent=a.get("prompt", ""),
                    brand=brand, product=(a.get("plan_chosen") or {}).get("form", ""),
                    model=edit_model if act == "edit" else gen_model,
                    method=f"{PILOT_LABELS.get(run.name, run.name)}; attempt {a['k']} ({act})",
                    run=run.name, seed=(a.get("made_by") or {}).get("seed", r.get("seed")),
                    clipscore=None, verify=v, judge=None), issues=rejudge(v, strat))


def from_flux_probe():
    from flux_probe import SCENES
    run = RUNS / "flux-probe"
    probe = json.loads((run / "probe.json").read_text())
    rv = json.loads((run / "reverify.json").read_text())
    for key, e in probe.items():
        s = SCENES[e["scene"]]
        for c, ce in e["conds"].items():
            img = run / key / f"{c}.png"
            v = rv.get(f"{key}/{c}")
            if not img.exists() or not v:
                continue
            yield dict(group_prompt=s["orig"], control=run / key / "B0_baseline.png", control_model=FLUX_DEV,
                       control_judge=None, inj=row(
                image=img, user_prompt=s["orig"], prompt_sent=ce["prompt"], brand=s["brand"], product=s["cls"],
                model=FLUX_DEV, method=f"FLUX placement probe, condition {c}", run="flux-probe", seed=e["seed"],
                clipscore=None, verify=v, judge=None), issues=rejudge(v))


def from_survey_closed():
    run = RUNS / "survey-closed"
    sel = {s["idx"]: s for s in json.loads((run / "selection.json").read_text())}
    for sp in sorted(run.glob("[0-9][0-9][0-9]/state.json")):
        st = json.loads(sp.read_text())
        d, brand = sp.parent, display(st["brand"])
        base_name = (sel.get(st["idx"]) or {}).get("base", "base")
        base = st["cond"]["base"] if base_name == "base" else \
            st["base_reroll"][int(base_name.split("_r")[1]) - 1]
        cands = [(k, c, SURVEY_CLOSED[k]) for k, c in st["cond"].items() if k in SURVEY_CLOSED]
        cands += [(f"loop{n + 1}", c, "rewriter v4 + judge-aware repair loop")
                  for n, c in enumerate(st.get("loop") or [])]
        for k, c, label in cands:
            if not (d / f"{k}.png").exists() or "verify" not in c:
                continue
            yield dict(group_prompt=st["prompt"], control=d / f"{base_name}.png", control_model=GEMINI_LITE,
                       control_judge=base.get("judge"), inj=row(
                image=d / f"{k}.png", user_prompt=st["prompt"], prompt_sent=c.get("prompt", ""), brand=brand,
                product=st.get("product", ""), model=GEMINI_LITE, method=label, run="survey-closed",
                seed=c.get("seed"), clipscore=c.get("clip"), verify=c["verify"], judge=c.get("judge")),
                issues=rejudge(c["verify"], (st["plan"].get("chosen") or {}).get("strategy", "")))


def from_survey_surface():
    run = RUNS / "survey-surface"
    for sp in sorted(run.glob("[0-9][0-9][0-9]/state.json")):
        st = json.loads(sp.read_text())
        d, brand = sp.parent, display(st["brand"])
        cands = [(k, c, SURVEY_SURFACE[k]) for k, c in st["cond"].items() if k in SURVEY_SURFACE]
        cands += [(f"loop{n + 1}", c, "surface-specific + feedback loop") for n, c in enumerate(st.get("loop") or [])]
        for k, c, label in cands:
            if not (d / f"{k}.png").exists() or "verify" not in c:
                continue
            yield dict(group_prompt=st["prompt"], control=d / "base.png", control_model=GEMINI_LITE,
                       control_judge=st["cond"]["base"].get("judge"), inj=row(
                image=d / f"{k}.png", user_prompt=st["prompt"], prompt_sent=c.get("prompt", ""), brand=brand,
                product=(st.get("phrase") or {}).get("product", ""), model=GEMINI_LITE, method=label,
                run="survey-surface", seed=c.get("seed"), clipscore=c.get("clip"), verify=c["verify"],
                judge=c.get("judge")), issues=rejudge(c["verify"], (st["plan"].get("chosen") or {}).get("strategy", "")))


# ------------------------------------------------------------------ export
def norm(p: str) -> str:
    return re.sub(r"^(a|an|the) ", "", re.sub(r"[^a-z0-9]+", " ", p.lower()).strip())


def digest(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()


def main():
    found = [*from_loop_runs(), *from_flux_probe(), *from_survey_closed(), *from_survey_surface()]
    total = len(found)
    keep, seen = [], set()
    for f in found:
        inj = f["inj"]
        if f["issues"] or not (ALL or "gemini" in inj["model"]):
            continue
        h = digest(inj["image"])
        if h in seen:  # survey-surface reuses some survey-closed renders
            continue
        seen.add(h)
        g = gold(f.get("_judge") or _judge_of(inj), inj["brand"])
        inj["tier"] = "gold" if g else "good"
        keep.append(f)

    groups: dict[str, str] = {}
    for f in keep:
        groups.setdefault(norm(f["group_prompt"]), f"G{len(groups) + 1:02d}")

    rows, controls_done = [], set()
    for f in keep:
        gid = groups[norm(f["group_prompt"])]
        ctl = f["control"]
        cj = f["control_judge"]
        if ctl and ctl.exists() and not (cj and cj.get("brands")):
            h = digest(ctl)
            if h not in controls_done:
                controls_done.add(h)
                rows.append(row(image=ctl, user_prompt=f["group_prompt"], prompt_sent=f["group_prompt"], brand="",
                                product="", model=f["control_model"], method="no-inject (original prompt)",
                                run=f["inj"]["run"], seed=f["inj"]["seed"], clipscore=None, verify=None, judge=cj)
                            | {"tier": "control", "condition": "control", "group_id": gid})
        rows.append(f["inj"] | {"condition": "injected", "group_id": gid})

    tier_order = {"gold": 0, "good": 1, "control": 2}
    rows.sort(key=lambda r: (r["group_id"], tier_order[r["tier"]], r["run"], str(r["method"])))

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    for old in (OUT / "images").glob("*.jpg"):
        old.unlink()
    for n, r in enumerate(rows, 1):
        r["item_id"] = f"I{n:04d}"
        r["source_image"] = str(Path(r.pop("image")).relative_to(ROOT))
        r["file"] = f"{r['item_id']}.jpg"
        r["open_weight_model"] = r["model"] in OPEN
        im = Image.open(ROOT / r["source_image"]).convert("RGB")
        r["width"], r["height"] = im.size
        im.save(OUT / "images" / r["file"], quality=90)

    cols = ["item_id", "file", "group_id", "condition", "tier", "user_prompt", "prompt_sent", "brand", "product",
            "model", "open_weight_model", "width", "height", "method", "run", "seed", "legibility", "area", "centrality", "depth_plane",
            "draws_attention", "integration", "blind_ad_suspicion", "blind_brands_seen", "gpt_ad_likelihood",
            "gpt_adherence", "gpt_brands", "gpt_brand_naturalness", "clipscore", "blind_out_of_place", "source_image"]
    with open(OUT / "key.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (OUT / "key.json").write_text(json.dumps([{c: r.get(c) for c in cols} for r in rows], indent=1))
    with open(OUT / "items.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "image", "prompt"])
        w.writerows([r["item_id"], f"images/{r['file']}", r["user_prompt"]] for r in rows)
    write_gallery(rows)

    # best.csv: per prompt, the most subtle injected image (gold first, then lowest GPT ad-likelihood,
    # then lowest blind ad-suspicion) plus that prompt's control, i.e. one balanced pair per prompt
    def subtle(r):
        return ({"gold": 0, "good": 1}[r["tier"]], r["gpt_ad_likelihood"] or 5, r["blind_ad_suspicion"] or 5)
    best = []
    for gid in sorted({r["group_id"] for r in rows}):
        g = [r for r in rows if r["group_id"] == gid]
        ctl = [r for r in g if r["condition"] == "control"]
        inj_g = sorted((r for r in g if r["condition"] == "injected"), key=subtle)
        if ctl and inj_g:
            pick = inj_g[0]
            # same run as the injected image if possible, else at least the same aspect ratio (a cue otherwise)
            same = [c for c in ctl if c["run"] == pick["run"]] or \
                   [c for c in ctl if (c["width"], c["height"]) == (pick["width"], pick["height"])]
            if same:
                best += [same[0], pick]
    with open(OUT / "best.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(best)

    inj = [r for r in rows if r["condition"] == "injected"]
    stats = {
        "injected_images_scanned": total,
        "injected_kept": len(inj),
        "gold": sum(r["tier"] == "gold" for r in inj),
        "good": sum(r["tier"] == "good" for r in inj),
        "controls": sum(r["condition"] == "control" for r in rows),
        "prompt_groups": len(groups),
        "open_weight_injected": sum(r["open_weight_model"] for r in inj),
        "by_run": {k: sum(r["run"] == k for r in inj) for k in sorted({r["run"] for r in inj})},
        "brands": {k: sum(r["brand"] == k for r in inj) for k in sorted({r["brand"] for r in inj})},
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


def _judge_of(inj: dict) -> dict | None:
    if inj.get("gpt_ad_likelihood") is None:
        return None
    return {"ad_likelihood": inj["gpt_ad_likelihood"], "adherence": inj["gpt_adherence"],
            "brands": [b for b in (inj["gpt_brands"] or "").split("; ") if b]}


def write_gallery(rows: list[dict]):
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["group_id"], []).append(r)
    parts = []
    for gid, rs in by.items():
        cards = "".join(
            f'<figure class="{r["tier"]}"><a href="images/{r["file"]}"><img loading="lazy" src="images/{r["file"]}"></a>'
            f'<figcaption><b>{r["item_id"]}</b> <span class="t">{r["tier"]}</span> {html.escape(r["brand"] or "")}'
            f'<br><small>{html.escape(r["method"])} · {html.escape(r["model"].split(" (")[0].split("/")[-1])}</small>'
            f'<br><small>sent: {html.escape(r["prompt_sent"])}</small></figcaption></figure>' for r in rs)
        parts.append(f'<section><h2>{gid} · {html.escape(rs[0]["user_prompt"])}</h2><div class="g">{cards}</div></section>')
    (OUT / "gallery.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Study pairs</title><style>"
        "body{font:14px system-ui;margin:16px;background:#fafafa}h2{font-size:15px;margin:24px 0 8px}"
        ".g{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px}"
        "figure{margin:0;background:#fff;border:2px solid #ddd;border-radius:6px;padding:6px}"
        "figure.gold{border-color:#c9a227}figure.good{border-color:#4a8f5b}figure.control{border-color:#999}"
        "img{width:100%;border-radius:4px}.t{text-transform:uppercase;font-size:11px;color:#666}</style>"
        f"<h1>Study text-image pairs ({len(rows)} images)</h1>" + "".join(parts))


if __name__ == "__main__":
    main()
