"""Build the human-judging lightbox (runs/lightbox/): every injected image with its same-seed
baseline, the prompt diff, and two automated verdicts (strict subtle band vs session-2 natural band),
revealed after the human votes. Labels are stored in the artifact's db (collection "labels").

    .venv/bin/python scripts/build_judge.py            # then publish runs/lightbox/index.html with img/*
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from adplace.__main__ import load_cfg  # noqa: E402
from adplace.pipeline import inserted_text, load_catalog  # noqa: E402
from adplace.verify import judge  # noqa: E402

OUT = ROOT / "runs" / "lightbox"
SHORT = {"A vintage-style painting of a family having a picnic on the beach": "Picnic painting",
         "a firefighter cooking a barbecue in a backyard with friends, summer afternoon, photo": "Firefighter BBQ",
         "A 1950s American diner with period-accurate tables, signs and chrome stools": "1950s diner",
         "an office desk at night with city lights through the window, cinematic lighting": "Office desk",
         "road trip, view from the back seat of a car driving through the desert at golden hour": "Road trip",
         "college students studying together in a dorm room late at night, warm lamp light, candid photo": "Dorm study",
         "friends watching a football match on a big screen at a rooftop bar at sunset": "Rooftop bar",
         "dystopian highly detailed city at night, with volumetric lights in the distance": "Dystopian city"}
COND_LABEL = {"C1_bare": "C1 · product only, no location", "C2_loc": "C2 · + location words",
              "C3_loc_size": "C3 · + location, 'small', 'out of focus'", "C4_generic": "C4 · C2 without the brand",
              "C5_front": "C5 · C2 phrase moved to the front", "C6_json": "C6 · FLUX.2 JSON prompt",
              "C7_inventory": "C7 · baseline props listed, product mid-list",
              "C8_crowd": "C8 · one among a crowd of drinks/objects"}


def jpg(src: Path) -> str:
    h = hashlib.sha1(str(src.resolve()).encode()).hexdigest()[:12]
    dst = OUT / "img" / f"{h}.jpg"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.open(src).convert("RGB").resize((896, 896), Image.LANCZOS).save(dst, quality=80, optimize=True)
    return f"img/{h}.jpg"


def verdicts(v: dict, cfg: dict, strategy: str = "") -> dict:
    p = cfg["prominence"]
    strict = judge(v["informed"], v["blind"], v["metrics"], p["subtle"], 4, 4, strategy, max_scene_loss=2.0)
    natural = judge(v["informed"], v["blind"], v["metrics"], p["natural"], 4, 4, strategy,
                    max_scene_loss=float(p.get("max_scene_loss", 0.5)))
    fmt = lambda iss: {"pass": not iss, "codes": [i["code"] for i in iss], "why": [i["detail"] for i in iss]}
    return {"strict": fmt(strict), "natural": fmt(natural)}


def facts(v: dict) -> dict:
    inf, b, m = v["informed"], v["blind"] or {}, v["metrics"]
    return {"area": m.get("area"), "count": m.get("count"), "leg": inf.get("legibility"),
            "att": inf.get("draws_attention"), "susp": b.get("ad_suspicion"),
            "seen": b.get("brands_seen") or [], "desc": b.get("description", ""),
            "odd": b.get("out_of_place") or [], "ref": inf.get("reference_objects") or [],
            "gone": inf.get("missing_vs_reference") or [], "fid": inf.get("fidelity_to_original"),
            "integ": inf.get("integration")}


def loop_items(run: Path, set_id: str, set_name: str, cfg: dict, cat: dict) -> list[dict]:
    rv = json.loads((run / "reverify.json").read_text()) if (run / "reverify.json").exists() else {}
    items = []
    for rp in sorted(run.glob("[0-9][0-9][0-9]/record.json")):
        r = json.loads(rp.read_text())
        if not r.get("attempts") or not r.get("baseline"):
            continue
        base = jpg(rp.parent / r["baseline"])
        for a in r["attempts"]:
            key = f"{rp.parent.name}/{a['k']}"
            v = rv.get(key) or (a["verify"] if a["verify"].get("informed") else None)
            if not v:
                continue
            items.append({
                "id": f"{set_id}-{rp.parent.name}-{a['k']}", "set": set_id, "setName": set_name,
                "group": SHORT.get(r["prompt"], r["prompt"][:40]), "label": f"attempt {a['k']}"
                + (" · final" if r.get("final") == a["image"] and r["status"] == "accepted" else ""),
                "brand": cat[r["brand"]]["display"], "orig": r["prompt"], "sent": a["prompt"],
                "ins": inserted_text(r["prompt"], a["prompt"]), "base": base, "img": jpg(rp.parent / a["image"]),
                "auto": verdicts(v, cfg, (a.get("plan_chosen") or {}).get("strategy", "")), "f": facts(v),
                "loop_status": r["status"]})
    return items


def probe_items(run: Path, cfg: dict) -> list[dict]:
    from flux_probe import SCENES
    rv = json.loads((run / "reverify.json").read_text()) if (run / "reverify.json").exists() else {}
    db = json.loads((run / "probe.json").read_text())
    names = {"rooftop": "Rooftop bar", "desk": "Office desk", "bbq": "Firefighter BBQ"}
    items = []
    for key, e in db.items():
        s = SCENES[e["scene"]]
        for c, ce in e["conds"].items():
            v = rv.get(f"{key}/{c}")
            img = run / key / f"{c}.png"
            if not v or not img.exists():
                continue
            items.append({
                "id": f"P-{e['scene']}-{e['seed']}-{c}", "set": "P", "setName": "FLUX probe",
                "group": names[e["scene"]], "label": f"{COND_LABEL[c]} · seed {e['seed']}", "cond": c,
                "brand": s["brand"], "orig": s["orig"], "sent": ce["prompt"],
                "ins": inserted_text(s["orig"], ce["prompt"]) if c != "C6_json" else "(JSON prompt)",
                "base": jpg(run / key / "B0_baseline.png"), "img": jpg(img),
                "auto": verdicts(v, cfg), "f": facts(v)})
    return items


def main():
    cfg = load_cfg(None)
    cat = load_catalog()
    items = []
    sets = [("V3", "runs/pilot-v3", "v3 rewriter (this session)"),
            ("G", "runs/pilot-v2-regen", "v2 + prompt repair (last pilot)")]
    for sid, path, name in sets:
        if (ROOT / path).exists():
            items += loop_items(ROOT / path, sid, name, cfg, cat)
    if (ROOT / "runs/flux-probe/probe.json").exists():
        items += probe_items(ROOT / "runs/flux-probe", cfg)
    tpl = (ROOT / "scripts" / "lightbox_template.html").read_text()
    html = tpl.replace("/*__ITEMS__*/[]", json.dumps(items, separators=(",", ":")))
    (OUT / "index.html").write_text(html)
    used = {i["img"] for i in items} | {i["base"] for i in items}
    print(f"{len(items)} items, {len(used)} images -> {OUT / 'index.html'}")
    (OUT / "files.json").write_text(json.dumps({u: str(OUT / u) for u in sorted(used)}))


if __name__ == "__main__":
    main()
