"""Fill docs/flux_limits.html with the per-scene probe numbers from runs/flux-probe/probe.json.
    .venv/bin/python docs/build_flux_limits.py   -> docs/flux_limits.built.html"""
import json
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent.parent
db = json.loads((ROOT / "runs/flux-probe/probe.json").read_text())
D = {}
for key, e in db.items():
    for c, ce in e["conds"].items():
        D.setdefault(e["scene"], {}).setdefault(c, []).append(ce["m"])
out = {}
for sc, conds in D.items():
    for c, ms in conds.items():
        ref = sum(len(m.get("reference_objects") or []) for m in ms)
        out.setdefault(sc, {})[c] = {
            "loc_yes": sum(m.get("location_followed") == "yes" for m in ms),
            "fg": sum(m.get("depth_plane") == "foreground" for m in ms),
            "area": median(m["area"] for m in ms),
            "att": mean(int(m.get("draws_attention") or 1) for m in ms),
            "susp": mean(int(m["blind"].get("ad_suspicion") or 1) for m in ms),
            "lost": sum(len(m.get("missing_in_B") or []) for m in ms) / max(1, ref),
            "comp": mean(int(m.get("composition_similarity") or 0) for m in ms),
        }
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from flux_instructions import INSTRUCTIONS, SEEDS
ires = json.loads((ROOT / "runs/flux-instructions/results.json").read_text())
NOTES = {"I06": "(Counted as followed, but the bottle is pasted onto the frame edge like a sticker.)",
         "I08": "(The bottle did come out smaller than in other conditions, just never smaller than an apple.)"}
instr = []
for iid, sc, cat, text in INSTRUCTIONS:
    rs = [ires[f"{iid}/{s}"] for s in SEEDS if f"{iid}/{s}" in ires]
    inst = next((r.get("what_it_did_instead") for r in rs if r.get("overall") != "yes" and r.get("what_it_did_instead")), "")
    instr.append({"id": iid, "instruction": text, "category": cat, "verdicts": [r["overall"] for r in rs],
                  "prom": mean(r.get("prominence", 0) for r in rs), "instead": inst, "note": NOTES.get(iid, "")})
src = (ROOT / "docs/flux_limits.html").read_text()
src = src.replace("/*__PROBE__*/{}", json.dumps(out)).replace("/*__INSTR__*/[]", json.dumps(instr))
(ROOT / "docs/flux_limits.built.html").write_text(src)
print("ok")
