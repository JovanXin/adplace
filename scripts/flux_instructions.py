"""FLUX.2 instruction-compliance test: does it do what a *specific* placement instruction says?

Each instruction is appended to an original pilot prompt (", <instruction>") and rendered on 4 seeds,
paired with the same-seed baseline. A VLM splits the instruction into atomic requirements and
judges each one (prompts/instruction_check.md). Categories cover containment/occlusion, support,
being held, pose/state, frame position, orientation, relative size, count and negation.

    .venv/bin/python scripts/flux_instructions.py            # generate + check (resumable)
    .venv/bin/python scripts/flux_instructions.py --analyse
Outputs: runs/flux-instructions/<scene>/<seed>/<id>.png, results.json
The BBQ and desk baselines are shared with runs/flux-probe (same prompts and seeds).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adplace.__main__ import load_cfg, make_llm, resolve_models  # noqa: E402
from adplace.generate import ComfyCloud  # noqa: E402
from adplace.llm import load_env  # noqa: E402
from adplace.pipeline import load_prompt  # noqa: E402

OUT = ROOT / "runs" / "flux-instructions"
PROBE = ROOT / "runs" / "flux-probe"
SEEDS = [5491897, 1234, 777, 42]
SCENES = {
    "picnic": "A vintage-style painting of a family having a picnic on the beach",
    "bbq": "a firefighter cooking a barbecue in a backyard with friends, summer afternoon, photo",
    "desk": "an office desk at night with city lights through the window, cinematic lighting",
}
# (id, scene, category, instruction)
INSTRUCTIONS = [
    ("I01", "picnic", "containment", "a Coca-Cola bottle half hidden inside the picnic basket"),
    ("I02", "picnic", "containment", "the top of a Coca-Cola bottle peeking out of the picnic basket"),
    ("I03", "picnic", "occlusion", "a Coca-Cola bottle mostly hidden behind the picnic basket"),
    ("I04", "picnic", "held", "the little girl holding a Coca-Cola bottle"),
    ("I05", "picnic", "pose/state", "a Coca-Cola bottle lying on its side on the blanket"),
    ("I06", "picnic", "frame position", "a Coca-Cola bottle at the far left edge of the picture"),
    ("I07", "picnic", "orientation", "a Coca-Cola bottle with its label turned away from the viewer"),
    ("I08", "picnic", "relative size", "a tiny Coca-Cola bottle, smaller than an apple"),
    ("I09", "bbq", "support", "a Coca-Cola can on top of the cooler"),
    ("I10", "bbq", "held", "a friend on the right holding a Coca-Cola can"),
    ("I11", "bbq", "pose/state", "a crushed Coca-Cola can lying in the grass"),
    ("I12", "bbq", "count", "exactly two Coca-Cola cans on the table"),
    ("I13", "desk", "occlusion", "a closed Apple MacBook half covered by a stack of papers"),
    ("I14", "desk", "pose/state", "an Apple iPhone lying face down next to a coffee mug"),
    ("I15", "desk", "frame position", "an Apple MacBook on the right side of the desk"),
    ("I16", "desk", "negation", "an Apple MacBook with no logo visible"),
]


def base_path(scene: str, seed: int) -> Path:
    p = OUT / scene / str(seed) / "B0_baseline.png"
    src = PROBE / scene / str(seed) / "B0_baseline.png"
    if not p.exists() and src.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse", action="store_true")
    args = ap.parse_args()
    if args.analyse:
        return analyse()
    load_env(ROOT / ".env")
    cfg = load_cfg(None)
    cfg["budget"]["max_image_generations"] = 100
    llm = make_llm(cfg, False, run_id="flux-instructions")
    vlm = resolve_models(cfg, llm)["verifier"]
    g = cfg["generation"]
    gen = ComfyCloud(os.environ["COMFY_API_KEY"], g["comfy_base_url"], str(ROOT / g["comfy_workflow"]), llm.ledger)
    OUT.mkdir(parents=True, exist_ok=True)
    rp = OUT / "results.json"
    res = json.loads(rp.read_text()) if rp.exists() else {}
    lock = threading.Lock()

    def save():
        with lock:
            rp.write_text(json.dumps(res, indent=1))

    jobs = []
    for sc in SCENES:
        for seed in SEEDS:
            b = base_path(sc, seed)
            if not b.exists():
                jobs.append((b, SCENES[sc], seed))
    for iid, sc, cat, instr in INSTRUCTIONS:
        for seed in SEEDS:
            jobs.append((OUT / sc / str(seed) / f"{iid}.png", f"{SCENES[sc]}, {instr}", seed))

    def gen_one(path, prompt, seed):
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gen.generate(prompt, seed, 1024, 1024))
        print("  gen", path.relative_to(OUT), flush=True)

    with ThreadPoolExecutor(2) as ex:
        list(ex.map(lambda j: gen_one(*j), jobs))

    def check(iid, sc, cat, instr, seed):
        key = f"{iid}/{seed}"
        if key in res:
            return
        img = (OUT / sc / str(seed) / f"{iid}.png").read_bytes()
        base = base_path(sc, seed).read_bytes()
        ctx = {"original_prompt": SCENES[sc], "instruction_appended": instr,
               "full_prompt_sent": f"{SCENES[sc]}, {instr}"}
        out = llm.chat_json(vlm, load_prompt("instruction_check"), ctx, task="instruction_check",
                            temperature=0.0, images=[base, img])
        res[key] = {"id": iid, "scene": sc, "category": cat, "instruction": instr, "seed": seed,
                    "prompt": ctx["full_prompt_sent"], **out}
        print(f"  {key}: {out.get('overall')}  {out.get('what_it_did_instead', '')[:90]}", flush=True)
        save()

    with ThreadPoolExecutor(6) as ex:
        for f in [ex.submit(check, iid, sc, cat, instr, seed)
                  for iid, sc, cat, instr in INSTRUCTIONS for seed in SEEDS]:
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                print("  check failed:", e, flush=True)
    save()
    print(f"ledger ${llm.ledger.spent:.3f}")
    analyse()


def analyse():
    res = json.loads((OUT / "results.json").read_text())
    by = defaultdict(list)
    for r in res.values():
        by[r["id"]].append(r)
    print(f"{'id':4s} {'category':15s} {'yes':>3s} {'part':>4s} {'no':>3s} {'count':>5s} {'prom':>4s}  instruction")
    for iid, sc, cat, instr in INSTRUCTIONS:
        rs = by.get(iid, [])
        c = Counter(r["overall"] for r in rs)
        print(f"{iid:4s} {cat:15s} {c['yes']:3d} {c['partly']:4d} {c['no']:3d} "
              f"{sum(r.get('count', 0) for r in rs) / max(1, len(rs)):5.1f} "
              f"{sum(r.get('prominence', 0) for r in rs) / max(1, len(rs)):4.1f}  {instr}")
    reqs = Counter()
    for r in res.values():
        for q in r.get("requirements", []):
            reqs[q.get("met")] += 1
    print("atomic requirements:", dict(reqs))


if __name__ == "__main__":
    main()
