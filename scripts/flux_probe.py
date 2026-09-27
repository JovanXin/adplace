"""FLUX.2 instruction-following probe: does the image model obey placement wording at all?

Controlled, paired design. For each scene x seed, one baseline (original prompt) and one image per
prompt condition, all on the same seed, so any difference is caused by the prompt text.

  C1_bare      orig + ", a Heineken bottle"                        (no location)
  C2_loc       orig + ", a Heineken bottle on a table in the far background"
  C3_loc_size  orig + ", a small Heineken bottle ... far background, softly out of focus"   (size + blur words)
  C4_generic   C2 with the brand removed ("a green beer bottle ...")  -> is the brand name the problem?
  C5_front     C2's phrase moved to the START of the prompt            -> word-order effect
  C6_json      FLUX.2 structured JSON prompt with a "position" field    (BFL-documented feature)
  C7_inventory orig + the props FLUX drew in the baseline, product in the middle of that list
               (baseline-anchored: keeps the clutter the product otherwise replaces)
  C8_crowd     orig + "a table in the background crowded with drinks, one of them a Heineken bottle"

Every image is measured against its baseline by a VLM (prompts/probe_measure.md: object boxes, depth,
focus, location compliance, which baseline props disappeared) plus the blind naive-viewer pass, and
a pixel metric (share of pixels changed vs the baseline, outside the product box).

    .venv/bin/python scripts/flux_probe.py [--seeds 3] [--measure-only]
    .venv/bin/python scripts/flux_probe.py --analyse
Outputs: runs/flux-probe/<scene>/<seed>/<cond>.png, runs/flux-probe/probe.json, analysis.json
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adplace.__main__ import load_cfg, make_llm, resolve_models  # noqa: E402
from adplace.generate import ComfyCloud  # noqa: E402
from adplace.llm import load_env  # noqa: E402
from adplace.pipeline import load_prompt  # noqa: E402
from adplace.verify import _norm_box, blind_pass, edit_drift  # noqa: E402

OUT = ROOT / "runs" / "flux-probe"
SEEDS = [5491897, 1234, 777, 42]

SCENES = {
    "rooftop": {
        "orig": "friends watching a football match on a big screen at a rooftop bar at sunset",
        "brand": "Heineken", "cls": "beer bottle",
        "b_obj": "a Heineken bottle", "g_obj": "a green beer bottle",
        "loc": "on a table in the far background",
        "json_main": {"description": "a group of friends watching a football match", "position": "foreground and midground, facing a big outdoor screen"},
        "json_scene": "a rooftop bar at sunset with a big screen showing a football match",
        "crowd": "a table in the far background crowded with drinks and glasses, one of them {b}",
    },
    "desk": {
        "orig": "an office desk at night with city lights through the window, cinematic lighting",
        "brand": "Apple", "cls": "laptop",
        "b_obj": "an Apple MacBook", "g_obj": "a silver laptop",
        "loc": "at the far back corner of the desk",
        "json_main": {"description": "an office desk at night", "position": "centre and foreground"},
        "json_scene": "an office at night, city lights through the window, cinematic lighting",
        "crowd": "the desk cluttered with papers, books and office supplies, among them {b}",
    },
    "bbq": {
        "orig": "a firefighter cooking a barbecue in a backyard with friends, summer afternoon, photo",
        "brand": "Coca-Cola", "cls": "soda can or bottle",
        "b_obj": "a Coca-Cola can", "g_obj": "a red soda can",
        "loc": "on a table in the far background",
        "json_main": {"description": "a firefighter cooking at a barbecue grill", "position": "centre foreground"},
        "json_scene": "a backyard barbecue with friends on a summer afternoon, photo",
        "crowd": "a table in the far background crowded with drinks and snacks, one of them {b}",
    },
}
CONDS = ["C1_bare", "C2_loc", "C3_loc_size", "C4_generic", "C5_front", "C6_json", "C7_inventory", "C8_crowd"]
LOC_CONDS = {"C2_loc", "C3_loc_size", "C4_generic", "C5_front", "C6_json", "C8_crowd"}

INVENTORY_SYS = (
    "List the notable props and objects visible in this image (not people, not walls/windows/sky), "
    "most noticeable first, as short plain noun phrases with an article ('a brass desk lamp', "
    "'an open notebook', 'a stack of papers'). Up to 6. Reply with JSON only: {\"props\": []}")


def build_prompt(s: dict, cond: str, props: list[str]) -> str:
    o, b, loc = s["orig"], s["b_obj"], s["loc"]
    if cond == "C1_bare":
        return f"{o}, {b}"
    if cond == "C2_loc":
        return f"{o}, {b} {loc}"
    if cond == "C3_loc_size":
        return f"{o}, a small {b.split(' ', 1)[1]} {loc}, softly out of focus"
    if cond == "C4_generic":
        return f"{o}, {s['g_obj']} {loc}"
    if cond == "C5_front":
        return f"{b[0].upper()}{b[1:]} {loc}; {o}"
    if cond == "C6_json":
        return json.dumps({
            "scene": s["json_scene"],
            "subjects": [{**s["json_main"], "action": ""},
                         {"description": b, "position": f"{loc}, small, far background", "action": "incidental prop"}],
            "style": "photo" if "photo" in o or "cinematic" in o or "rooftop" in o else "photo",
            "composition": "the main subject dominates the frame; the second subject is a tiny background detail",
            "camera": {"depth_of_field": "focus on the main subject, background soft"},
        })
    if cond == "C7_inventory":
        items = list(props[:4])
        items.insert(min(2, len(items)), b)     # product mid-list, never first or last
        return f"{o}, with {', '.join(items[:-1])} and {items[-1]}"
    if cond == "C8_crowd":
        return f"{o}, {s['crowd'].format(b=b)}"
    raise ValueError(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--measure-only", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    args = ap.parse_args()
    if args.analyse:
        return analyse()
    load_env(ROOT / ".env")
    cfg = load_cfg(None)
    cfg["budget"]["max_image_generations"] = 12 * 9 + 20   # probe needs 108 FLUX images (Comfy credits, not OpenRouter)
    import os
    llm = make_llm(cfg, False, run_id="flux-probe")
    vlm = resolve_models(cfg, llm)["verifier"]
    g = cfg["generation"]
    gen = ComfyCloud(os.environ["COMFY_API_KEY"], g["comfy_base_url"], str(ROOT / g["comfy_workflow"]), llm.ledger)
    OUT.mkdir(parents=True, exist_ok=True)
    dbp = OUT / "probe.json"
    db = json.loads(dbp.read_text()) if dbp.exists() else {}
    lock = threading.Lock()

    def save():
        with lock:
            dbp.write_text(json.dumps(db, indent=1))

    scenes = args.scenes.split(",")
    seeds = SEEDS[:args.seeds]

    def gen_one(path: Path, prompt: str, seed: int):
        if not path.exists() and not args.measure_only:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(gen.generate(prompt, seed, 1024, 1024))
            print("  gen", path.relative_to(OUT), flush=True)

    # 1. baselines + their prop inventory (used by C7)
    for sc in scenes:
        for seed in seeds:
            key = f"{sc}/{seed}"
            gen_one(OUT / sc / str(seed) / "B0_baseline.png", SCENES[sc]["orig"], seed)
            e = db.setdefault(key, {"scene": sc, "seed": seed, "conds": {}})
            if "props" not in e:
                inv = llm.chat_json(vlm, INVENTORY_SYS, f"Prompt: {SCENES[sc]['orig']!r}", task="probe_inventory",
                                    temperature=0.0, images=[(OUT / sc / str(seed) / "B0_baseline.png").read_bytes()])
                e["props"] = [p for p in inv.get("props", []) if isinstance(p, str)][:6]
                save()

    # 2. conditions (Comfy runs one job at a time; 2 threads keep the queue fed)
    jobs = []
    for sc in scenes:
        for seed in seeds:
            e = db[f"{sc}/{seed}"]
            for c in CONDS:
                p = build_prompt(SCENES[sc], c, e["props"])
                e["conds"].setdefault(c, {})["prompt"] = p
                jobs.append((OUT / sc / str(seed) / f"{c}.png", p, seed))
    save()
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(lambda j: gen_one(*j), jobs))

    # 3. measure
    def measure(sc, seed, c):
        e = db[f"{sc}/{seed}"]["conds"][c]
        if "m" in e:
            return
        s = SCENES[sc]
        base = (OUT / sc / str(seed) / "B0_baseline.png").read_bytes()
        img = (OUT / sc / str(seed) / f"{c}.png").read_bytes()
        ctx = {"original_prompt": s["orig"], "modified_prompt": e["prompt"], "object_class": s["cls"],
               "brand": s["brand"], "location_asked": s["loc"] if c in LOC_CONDS else "none"}
        m = llm.chat_json(vlm, load_prompt("probe_measure"), ctx, task="probe_measure", temperature=0.0,
                          images=[base, img])
        bl = blind_pass(llm, vlm, img, s["orig"])
        inst = m.get("instances") or []
        ti = m.get("target_index")
        tgt = inst[ti] if isinstance(ti, int) and 0 <= ti < len(inst) else (inst[0] if inst else None)
        nb = _norm_box(tgt["box"]) if tgt else None
        area = (nb[2] - nb[0]) * (nb[3] - nb[1]) if nb else 0.0
        drift = edit_drift(base, img, [nb] if nb else [], pad=0.02)
        e["m"] = {**m, "blind": bl, "target_box": nb, "area": round(area, 5),
                  "bottom_y": round(nb[2], 3) if nb else None,
                  "n_instances": len(inst), "n_branded": sum(1 for i in inst if i.get("brand_visible")),
                  "pixel_change": drift}
        print(f"  measured {sc}/{seed}/{c}: area={area:.4f} depth={m.get('depth_plane')} "
              f"loc={m.get('location_followed')} missing={len(m.get('missing_in_B') or [])} susp={bl.get('ad_suspicion')}", flush=True)
        save()

    tasks = [(sc, seed, c) for sc in scenes for seed in seeds for c in CONDS
             if (OUT / sc / str(seed) / f"{c}.png").exists()]
    with ThreadPoolExecutor(6) as ex:
        for f in [ex.submit(measure, *t) for t in tasks]:
            try:
                f.result()
            except Exception as err:  # noqa: BLE001
                print("  measure failed:", err, flush=True)
    print(f"OpenRouter ledger total ${llm.ledger.spent:.3f}")
    analyse()


def analyse():
    from statistics import mean, median
    db = json.loads((OUT / "probe.json").read_text())
    rows = {}
    for key, e in db.items():
        for c, ce in e["conds"].items():
            if "m" in ce:
                rows.setdefault(c, []).append((e["scene"], ce["m"]))
    out = {}
    for c in CONDS:
        rs = [m for _, m in rows.get(c, [])]
        if not rs:
            continue
        n = len(rs)
        found = [m for m in rs if m["n_instances"]]
        props = [len(m.get("reference_objects") or []) for m in rs]
        miss = [len(m.get("missing_in_B") or []) for m in rs]
        out[c] = {
            "n": n,
            "found": round(len(found) / n, 2),
            "brand_identifiable": round(sum(1 for m in rs if m["n_branded"]) / n, 2),
            "median_area": round(median([m["area"] for m in found]), 4) if found else None,
            "median_bottom_y": round(median([m["bottom_y"] for m in found if m["bottom_y"] is not None]), 3) if found else None,
            "foreground": round(sum(1 for m in found if m.get("depth_plane") == "foreground") / max(1, len(found)), 2),
            "background": round(sum(1 for m in found if m.get("depth_plane") == "background") / max(1, len(found)), 2),
            "in_focus": round(sum(1 for m in found if m.get("in_focus")) / max(1, len(found)), 2),
            "location_yes": round(sum(1 for m in rs if m.get("location_followed") == "yes") / n, 2) if c in LOC_CONDS else None,
            "location_yes_or_partly": round(sum(1 for m in rs if m.get("location_followed") in ("yes", "partly")) / n, 2) if c in LOC_CONDS else None,
            "mean_draws_attention": round(mean([int(m.get("draws_attention") or 1) for m in found]), 2) if found else None,
            "mean_legibility": round(mean([int(m.get("legibility") or 0) for m in found]), 2) if found else None,
            "mean_blind_suspicion": round(mean([int(m["blind"].get("ad_suspicion") or 1) for m in rs]), 2),
            "props_lost_share": round(sum(miss) / max(1, sum(props)), 2),
            "mean_composition_similarity": round(mean([int(m.get("composition_similarity") or 0) for m in rs]), 2),
            "mean_subject_preserved": round(mean([int(m.get("subject_preserved") or 0) for m in rs]), 2),
            "median_pixel_change": round(median([m["pixel_change"] for m in rs]), 3),
            "on_main_subject": round(sum(1 for m in found if m.get("on_main_subject")) / max(1, len(found)), 2),
        }
    (OUT / "analysis.json").write_text(json.dumps(out, indent=1))
    cols = ["n", "found", "brand_identifiable", "median_area", "median_bottom_y", "foreground", "background",
            "in_focus", "location_yes", "mean_draws_attention", "mean_legibility", "mean_blind_suspicion",
            "props_lost_share", "mean_composition_similarity", "median_pixel_change"]
    print("cond          " + " ".join(f"{c[:9]:>9s}" for c in cols))
    for c, v in out.items():
        print(f"{c:13s} " + " ".join(f"{str(v.get(k)):>9s}" for k in cols))
    return out


if __name__ == "__main__":
    main()
