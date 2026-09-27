"""Build the project website (runs/site/): a long-form story page plus a data room.

Collects every result so far into data.json (pilot setups, classifier decisions, the wording probe,
the instruction test, every loop attempt of G and v3, Jovan's lightbox votes, both automated judges,
spend, and the LLM stage prompts), packs the images into tile sprites (the artifact allows at most
255 files), and fills scripts/site_template.html.

    .venv/bin/python scripts/build_site.py LABELS_DIR
LABELS_DIR = a folder of the lightbox's `labels` documents (ArtifactData list with out_dir).
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "docs"))

from adplace.__main__ import load_cfg  # noqa: E402
from adplace.pipeline import inserted_text, load_catalog  # noqa: E402
from build_judge import SHORT, facts, verdicts  # noqa: E402
from flux_instructions import INSTRUCTIONS, SEEDS, SCENES as ISCENES  # noqa: E402
from flux_probe import CONDS, SCENES as PSCENES  # noqa: E402

OUT = ROOT / "runs" / "site"
TILE = 400


class Sprites:
    def __init__(self):
        self.sheets = []   # {url, cols, rows}

    def sheet(self, name: str, grid: list[list[Path | None]]) -> list[list[dict | None]]:
        rows, cols = len(grid), max(len(r) for r in grid)
        im = Image.new("RGB", (cols * TILE, rows * TILE), (20, 24, 30))
        refs = []
        for r, row in enumerate(grid):
            rr = []
            for c, p in enumerate(row):
                if p and Path(p).exists():
                    im.paste(Image.open(p).convert("RGB").resize((TILE, TILE), Image.LANCZOS), (c * TILE, r * TILE))
                    rr.append({"s": len(self.sheets), "c": c, "r": r})
                else:
                    rr.append(None)
            refs.append(rr)
        url = f"img/sheet_{name}.jpg"
        im.save(OUT / url, quality=78, optimize=True, progressive=True)
        self.sheets.append({"url": url, "cols": cols, "rows": rows})
        return refs


def load_labels(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("*.json")):
        j = json.loads(f.read_text())
        out[f.stem] = j.get("data", j)
    return out


def kappa(pairs):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else None


def main(labels_dir: Path):
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "img").mkdir(parents=True)
    cfg = load_cfg(None)
    cat = load_catalog()
    L = load_labels(labels_dir)
    sp = Sprites()
    data: dict = {}

    # ---------------- pilot setups ----------------
    fs = json.loads((ROOT / "docs/final_stats.json").read_text())
    data["setups"] = [{"id": k, "label": v["label"], "n": v["n"], "accepted": v["accepted"], "variants": v["variants"],
                       "median_area": v["median_area"], "mean_leg": v["mean_leg"], "mean_susp": v["mean_susp"],
                       "brand_seen_pct": v["brand_seen_pct"], "clip_d": v["clip_d_att1"], "usd": v["usd"],
                       "usd_per_prompt": v["usd_per_prompt"], "images": v["n_images"], "codes": v["codes"]}
                      for k, v in fs.items() if isinstance(v, dict) and "label" in v]

    # ---------------- classifier ----------------
    cls = []
    for rp in sorted((ROOT / "runs/pilot-v3").glob("[0-9][0-9][0-9]/record.json")):
        r = json.loads(rp.read_text())
        c = r.get("classify") or {}
        b = (c.get("brands") or [{}])[0].get("brand_key") if c.get("decision") == "INJECT" else None
        cls.append({"idx": r["idx"], "prompt": r["prompt"], "decision": c.get("decision"),
                    "brand": cat[b]["display"] if b in cat else None, "reason": c.get("reason", ""),
                    "source": c.get("source", "llm"), "flags": c.get("exclude_flags") or []})
    data["classify"] = cls

    # ---------------- wording probe ----------------
    P = ROOT / "runs/flux-probe"
    pdb = json.loads((P / "probe.json").read_text())
    prv = json.loads((P / "reverify.json").read_text())
    probe = []
    for sc in PSCENES:
        grid = [[P / sc / str(seed) / f"{c}.png" for c in ["B0_baseline"] + CONDS] for seed in SEEDS]
        refs = sp.sheet(f"probe_{sc}", grid)
        for ri, seed in enumerate(SEEDS):
            e = pdb[f"{sc}/{seed}"]
            for ci, c in enumerate(CONDS, start=1):
                ce = e["conds"][c]
                m = ce.get("m") or {}
                v = prv.get(f"{sc}/{seed}/{c}")
                iid = f"P-{sc}-{seed}-{c}"
                auto = verdicts(v, cfg) if v else None
                f = facts(v) if v else {}
                probe.append({
                    "id": iid, "scene": sc, "seed": seed, "cond": c, "prompt": ce["prompt"], "orig": PSCENES[sc]["orig"],
                    "brand": PSCENES[sc]["brand"], "tile": refs[ri][ci], "base": refs[ri][0],
                    "area": m.get("area"), "depth": m.get("depth_plane"), "focus": m.get("in_focus"),
                    "loc": m.get("location_followed"), "loc_note": m.get("location_note"),
                    "leg": m.get("legibility"), "att": m.get("draws_attention"),
                    "susp": (m.get("blind") or {}).get("ad_suspicion"), "desc": (m.get("blind") or {}).get("description"),
                    "ref": m.get("reference_objects") or [], "gone": m.get("missing_in_B") or [],
                    "comp": m.get("composition_similarity"), "pix": m.get("pixel_change"),
                    "n_inst": m.get("n_instances"), "n_brand": m.get("n_branded"),
                    "box": m.get("target_box"),
                    "human": (L.get(iid) or {}).get("verdict"), "auto": auto,
                    "gone2": f.get("gone"), "ref2": f.get("ref")})
    data["probe"] = probe

    # ---------------- instruction test ----------------
    I = ROOT / "runs/flux-instructions"
    ires = json.loads((I / "results.json").read_text())
    instr = []
    for sc in ISCENES:
        ins = [x for x in INSTRUCTIONS if x[1] == sc]
        grid = [[I / sc / str(seed) / "B0_baseline.png"] + [I / sc / str(seed) / f"{iid}.png" for iid, *_ in ins] for seed in SEEDS]
        refs = sp.sheet(f"instr_{sc}", grid)
        for ri, seed in enumerate(SEEDS):
            for ci, (iid, _, cat_, text) in enumerate(ins, start=1):
                r = ires.get(f"{iid}/{seed}", {})
                instr.append({"id": f"{iid}-{seed}", "iid": iid, "scene": sc, "seed": seed, "category": cat_,
                              "instruction": text, "prompt": f"{ISCENES[sc]}, {text}", "orig": ISCENES[sc],
                              "overall": r.get("overall"), "requirements": r.get("requirements", []),
                              "instead": r.get("what_it_did_instead", ""), "count": r.get("count"),
                              "changed": r.get("scene_changed"), "prom": r.get("prominence"),
                              "tile": refs[ri][ci], "base": refs[ri][0]})
    data["instr"] = instr
    data["instrList"] = [{"iid": i, "scene": s, "category": c, "instruction": t} for i, s, c, t in INSTRUCTIONS]

    # ---------------- loop runs (G and v3) ----------------
    loops = []
    for sid, run, name in [("G", "pilot-v2-regen", "v2 rewriter + prompt repair"), ("V3", "pilot-v3", "v3 baseline-aware rewriter")]:
        d = ROOT / "runs" / run
        rv = json.loads((d / "reverify.json").read_text()) if (d / "reverify.json").exists() else {}
        recs = [json.loads(p.read_text()) for p in sorted(d.glob("[0-9][0-9][0-9]/record.json"))]
        recs = [r for r in recs if r.get("attempts") and r.get("baseline")]
        width = 1 + max(len(r["attempts"]) for r in recs)
        grid = [[d / f"{r['idx']:03d}" / r["baseline"]] + [d / f"{r['idx']:03d}" / a["image"] for a in r["attempts"]] for r in recs]
        refs = sp.sheet(f"loop_{sid.lower()}", [g + [None] * (width - len(g)) for g in grid])
        for ri, r in enumerate(recs):
            for a in r["attempts"]:
                key = f"{r['idx']:03d}/{a['k']}"
                v = rv.get(key) or (a["verify"] if a["verify"].get("informed") else None)
                iid = f"{sid}-{r['idx']:03d}-{a['k']}"
                inf = (v or {}).get("informed") or {}
                boxes = [[x / 1000 for x in b] for b in (inf.get("product_boxes") or []) if isinstance(b, list) and len(b) == 4]
                rep = a.get("repair") or {}
                loops.append({
                    "id": iid, "set": sid, "setName": name, "idx": r["idx"], "scene": SHORT.get(r["prompt"], r["prompt"][:30]),
                    "attempt": a["k"], "brand": cat[r["brand"]]["display"], "orig": r["prompt"], "prompt": a["prompt"],
                    "ins": inserted_text(r["prompt"], a["prompt"]), "tile": refs[ri][a["k"]], "base": refs[ri][0],
                    "boxes": boxes, "human": (L.get(iid) or {}).get("verdict"),
                    "auto": verdicts(v, cfg, (a.get("plan_chosen") or {}).get("strategy", "")) if v else None,
                    "f": facts(v) if v else None, "runtime_codes": a.get("codes"), "status": r["status"],
                    "feedback": rep.get("revision_feedback", ""), "anchor": (r.get("rewrite") or {}).get("anchor_in_baseline")})
    data["loops"] = loops

    # ---------------- agreement ----------------
    judged = [x for x in probe + loops if x.get("human") and x.get("auto")]
    ag = {}
    for judge in ["strict", "natural"]:
        for name, h in [("bpass", lambda v: v != "fail"), ("bfail", lambda v: v == "pass")]:
            pr = [(h(x["human"]), x["auto"][judge]["pass"]) for x in judged]
            ag[f"{judge}_{name}"] = {"agree": sum(a == b for a, b in pr) / len(pr), "kappa": kappa(pr),
                                     "lenient": sum((not a) and b for a, b in pr), "strict": sum(a and not b for a, b in pr)}
    susp5 = [x for x in judged if (x["auto"] and ((x.get("f") or {}).get("susp") or x.get("susp")) == 5)]
    data["agreement"] = {"n": len(judged), "humans": dict(Counter(x["human"] for x in judged)), **ag,
                         "recal": {"agree": 0.719, "kappa": 0.46, "note": "in-sample: blind ≤4, props lost <75%, no duplicate gate, no legibility 4"},
                         "susp5_pass": sum(1 for x in susp5 if x["human"] == "pass"), "susp5_n": len(susp5)}

    # ---------------- summaries ----------------
    ps = {}
    for sc in PSCENES:
        for c in CONDS:
            rows = [x for x in probe if x["scene"] == sc and x["cond"] == c]
            ref = sum(len(x["ref"]) for x in rows)
            ps.setdefault(sc, {})[c] = {
                "loc": sum(x["loc"] == "yes" for x in rows), "fg": sum(x["depth"] == "foreground" for x in rows),
                "area": median([x["area"] or 0 for x in rows]), "att": mean([int(x["att"] or 1) for x in rows]),
                "susp": mean([int(x["susp"] or 1) for x in rows]),
                "lost": sum(len(x["gone"]) for x in rows) / max(1, ref),
                "human": dict(Counter(x["human"] for x in rows))}
    data["probeSummary"] = ps
    data["condNames"] = {"C1_bare": "Product only", "C2_loc": "+ location words", "C3_loc_size": "+ location, “small”, “out of focus”",
                         "C4_generic": "Location words, no brand", "C5_front": "Phrase moved to the front", "C6_json": "FLUX.2 JSON prompt",
                         "C7_inventory": "Scene's own props listed", "C8_crowd": "One among a crowd"}
    data["sceneNames"] = {"rooftop": "Rooftop bar", "desk": "Office desk", "bbq": "Firefighter BBQ", "picnic": "Picnic painting"}
    hum = defaultdict(Counter)
    for x in loops:
        if x["human"]:
            hum[x["set"]][x["human"]] += 1
    data["loopHuman"] = {k: dict(v) for k, v in hum.items()}

    # ---------------- spend ----------------
    runs = defaultdict(float)
    tasks = defaultdict(float)
    total = 0.0
    for line in (ROOT / "runs/ledger.jsonl").read_text().splitlines():
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        c = float(j.get("cost") or 0)
        total += c
        runs[j.get("run_id") or "(setup)"] += c
        tasks[j.get("task", "?").split(":")[0]] += c
    data["spend"] = {"total": total, "runs": sorted(([k, v] for k, v in runs.items() if v > 0.0005), key=lambda x: -x[1]),
                     "tasks": sorted(([k, v] for k, v in tasks.items() if v > 0.0005), key=lambda x: -x[1])}

    # ---------------- prompts ----------------
    from build_prompt_appendix import STAGES, inline_prompts
    data["prompts"] = [{"key": k, "title": t, "model": m, "user": u, "text": (ROOT / "adplace/prompts" / f"{k}.md").read_text()}
                       for k, t, m, u in STAGES] + [{"key": k, "title": t, "model": m, "user": u, "text": b} for k, t, m, u, b in inline_prompts()]

    data["sheets"] = sp.sheets
    data["links"] = {"lightbox": "https://claude.ai/artifact/XuzJB2iBBwGK1biM6SJ7r3",
                     "flux": "https://claude.ai/artifact/4bYqS9zXCnE3VR5e7zbQr9",
                     "prompts": "https://claude.ai/artifact/Qx3b7dJzLnV3TVyeQCBrRe",
                     "deck": "https://claude.ai/artifact/A7XNheoRtkPBcJUFiSzrgU",
                     "pilot": "https://claude.ai/artifact/1EVuuaDTAx6CZyUNChFHL8",
                     "brief": "https://claude.ai/artifact/BBWcmuKbrXaczu3B3J1eWC"}
    (OUT / "data.json").write_text(json.dumps(data, separators=(",", ":"), default=str))

    # ---------------- story images ----------------
    deck = Path("/private/tmp/claude-501/-Users-jovanxin-Thesis-Claude/9601f092-90d8-48c3-807c-717267fbf6bc/scratchpad/deck")
    for sub in ("img", "img2"):
        for f in (deck / sub).glob("*.jpg"):
            shutil.copy(f, OUT / "img" / f.name)
    Image.open(ROOT / "runs/pilot-v3/006/attempt_1.png").convert("RGB").save(OUT / "img/hero.jpg", quality=84, optimize=True, progressive=True)
    Image.open(ROOT / "runs/pilot-v3/006/baseline.png").convert("RGB").save(OUT / "img/hero_base.jpg", quality=84, optimize=True, progressive=True)
    for f in ["F5", "E4", "bbq_v1", "bbq_v2", "bar_v1", "bar_v2", "bbq_base", "bar_base"]:
        src = ROOT / "docs/final_img" / f"{f}.jpg"
        if src.exists():
            shutil.copy(src, OUT / "img" / f"{f}.jpg")
    hero_box = next((x["boxes"] for x in loops if x["id"] == "V3-006-1"), [])

    tpl = (ROOT / "scripts/site_template.html").read_text()
    html = tpl.replace("/*__HEROBOX__*/null", json.dumps(hero_box))
    (OUT / "index.html").write_text(html)
    files = sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file() and p.name != "index.html")
    (OUT / "files.json").write_text(json.dumps(files))
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"{len(files)} files, {size / 1e6:.1f} MB; probe {len(probe)}, instr {len(instr)}, loops {len(loops)}, judged {len(judged)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
