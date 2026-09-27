"""Re-run the session-2 verifier (blind + two-image informed pass against the same-seed baseline)
over existing images, so every image in the judging lightbox has the same automated verdicts.

    .venv/bin/python scripts/reverify.py runs/pilot-v2-regen      # loop-format run
    .venv/bin/python scripts/reverify.py runs/flux-probe          # probe
Writes <dir>/reverify.json: {item_key: {"blind", "informed", "metrics"}}; the judges are applied
offline (scripts/build_judge.py) so bands can change without paying again.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from adplace.__main__ import load_cfg, make_llm, resolve_models  # noqa: E402
from adplace.llm import load_env  # noqa: E402
from adplace.pipeline import load_catalog  # noqa: E402
from adplace.verify import blind_pass, box_metrics, informed_pass  # noqa: E402


def jobs_for(run: Path):
    cat = load_catalog()
    if (run / "probe.json").exists():
        from flux_probe import SCENES
        db = json.loads((run / "probe.json").read_text())
        for key, e in db.items():
            s = SCENES[e["scene"]]
            base = run / key / "B0_baseline.png"
            for c, ce in e["conds"].items():
                img = run / key / f"{c}.png"
                if img.exists():
                    yield (f"{key}/{c}", img, base, s["orig"], ce["prompt"], s["brand"],
                           {"form": s["cls"], "anchor": s["loc"]})
        return
    for rp in sorted(run.glob("[0-9][0-9][0-9]/record.json")):
        r = json.loads(rp.read_text())
        if not r.get("attempts") or not r.get("baseline"):
            continue
        brand = cat[r["brand"]]["display"]
        for a in r["attempts"]:
            yield (f"{rp.parent.name}/{a['k']}", rp.parent / a["image"], rp.parent / r["baseline"],
                   r["prompt"], a["prompt"], brand, a.get("plan_chosen") or {})


def main(run: Path):
    load_env(ROOT / ".env")
    cfg = load_cfg(None)
    llm = make_llm(cfg, False, run_id=f"reverify-{run.name}")
    vlm = resolve_models(cfg, llm)["verifier"]
    outp = run / "reverify.json"
    out = json.loads(outp.read_text()) if outp.exists() else {}

    def one(job):
        key, img, base, orig, sent, brand, chosen = job
        if key in out:
            return
        im, bl = img.read_bytes(), base.read_bytes()
        blind = blind_pass(llm, vlm, im, orig)
        ctx = {"original_prompt": orig, "prompt_sent_to_image_model": sent,
               "intended_product": {"brand": brand, "form": chosen.get("form"), "anchor": chosen.get("anchor"),
                                    "spatial_relation": chosen.get("spatial_relation")}}
        inf = informed_pass(llm, vlm, im, ctx, reference=bl)
        out[key] = {"blind": blind, "informed": inf,
                    "metrics": box_metrics(inf.get("product_boxes"), inf.get("main_subject_box"))}
        print(f"  {key}: susp={blind.get('ad_suspicion')} leg={inf.get('legibility')} "
              f"gone={len(inf.get('missing_vs_reference') or [])}/{len(inf.get('reference_objects') or [])}", flush=True)

    with ThreadPoolExecutor(6) as ex:
        for i, f in enumerate([ex.submit(one, j) for j in jobs_for(run)]):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                print("  failed:", e, flush=True)
            if i % 10 == 9:
                outp.write_text(json.dumps(out, indent=1))
    outp.write_text(json.dumps(out, indent=1))
    print(f"{len(out)} items; ledger ${llm.ledger.spent:.3f}")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        main(Path(a).resolve())
