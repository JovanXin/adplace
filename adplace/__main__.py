"""CLI.  python -m adplace {models,check,classify,run,report} [--mock] ...

  models                         list OpenRouter models (flags image in/out)
  check                          key balance, resolved model ids, Comfy workflow auto-detect dry run
  classify FILE                  classify each prompt (gate + LLM), print a table
  run FILE [--limit N] [--dry-run] [--repair edit|regen] [--mode rewrite|vpp] [--workers N] [--name NAME]
  report RUN_DIR                 write RUN_DIR/report/index.html
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

from .llm import Ledger, OpenRouter, load_env
from .pipeline import classify, load_catalog

ROOT = Path(__file__).resolve().parent.parent


def load_cfg(path: str | None) -> dict:
    p = Path(path) if path else (ROOT / "config.yaml" if (ROOT / "config.yaml").exists() else ROOT / "config.example.yaml")
    return yaml.safe_load(p.read_text())


def read_prompts(path: str, limit: int | None = None) -> list[str]:
    lines = [l.strip() for l in Path(path).read_text().splitlines()]
    ps = [l for l in lines if l and not l.startswith("#")]
    return ps[:limit] if limit else ps


def make_llm(cfg: dict, mock: bool, run_id: str = ""):
    if mock:
        from .mock import MockLLM
        return MockLLM()
    b = cfg["budget"]
    ledger = Ledger(ROOT / "runs" / "ledger.jsonl", float(b["openrouter_usd_cap"]),
                    int(b["max_image_generations"]), run_id=run_id)
    return OpenRouter(os.environ.get("OPENROUTER_API_KEY", ""), ledger)


def resolve_models(cfg: dict, llm) -> dict:
    m = cfg["models"]
    return {"classifier": llm.resolve(m["classifier"]),
            "rewriter": llm.resolve(m["rewriter"]),
            "verifier": llm.resolve(m["verifier"], need_image_in=True),
            **({"repair": llm.resolve(m["repair"], need_image_in=True)} if m.get("repair") else {})}


def make_backends(cfg: dict, llm, mock: bool):
    g, e = cfg["generation"], cfg["edit"]
    if mock or g["backend"] == "mock":
        from .mock import MockImage
        mi = MockImage(llm.ledger)
        return mi, (mi if e["backend"] != "none" else None)
    from .generate import ComfyCloud, ComfyEdit, OpenRouterImage
    key = os.environ.get("COMFY_API_KEY", "")
    if g["backend"] == "comfy":
        gen = ComfyCloud(key, g["comfy_base_url"], str(ROOT / g["comfy_workflow"]), llm.ledger,
                         g.get("comfy_prompt_node"), g.get("comfy_seed_node"))
    else:
        gen = OpenRouterImage(llm, llm.resolve(g["openrouter_image_model"], need_image_out=True))
    if e["backend"] == "comfy":
        ed = ComfyEdit(key, g["comfy_base_url"], str(ROOT / e["comfy_edit_workflow"]), llm.ledger)
    elif e["backend"] == "openrouter":
        ed = OpenRouterImage(llm, llm.resolve(cfg["models"]["editor"], need_image_in=True, need_image_out=True))
    else:
        ed = None
    return gen, ed


def cmd_models(args, cfg):
    llm = make_llm(cfg, False)
    for m in sorted(llm.models(), key=lambda m: m["id"]):
        a = m.get("architecture", {})
        if args.filter and args.filter not in m["id"]:
            continue
        flags = ("img-in " if "image" in a.get("input_modalities", []) else "       ") + \
                ("img-out" if "image" in a.get("output_modalities", []) else "")
        p = m.get("pricing", {})
        print(f"{m['id']:60s} {flags:15s} in ${float(p.get('prompt') or 0) * 1e6:.2f}/M  out ${float(p.get('completion') or 0) * 1e6:.2f}/M")


def cmd_check(args, cfg):
    llm = make_llm(cfg, args.mock)
    if not args.mock:
        ks = llm.key_status()
        print(f"OpenRouter key: limit ${ks.get('limit')}  remaining ${ks.get('limit_remaining')}  used ${ks.get('usage')}")
        print(f"Ledger: ${llm.ledger.spent:.4f} spent of ${llm.ledger.cap_usd:.2f} cap")
    print("Models:", json.dumps(resolve_models(cfg, llm), indent=1))
    gen, ed = make_backends(cfg, llm, args.mock)
    for label, b in (("generate", gen), ("edit", ed)):
        if b is None:
            print(f"{label}: none")
        elif hasattr(b, "prompt_node"):
            print(f"{label}: Comfy workflow, prompt node {b.prompt_node}, seed node {b.seed_node}")
        else:
            print(f"{label}: {type(b).__name__}")


def cmd_classify(args, cfg):
    llm = make_llm(cfg, args.mock)
    model = resolve_models(cfg, llm)["classifier"]
    cat = load_catalog()
    for p in read_prompts(args.file, args.limit):
        c = classify(llm, model, p, cat)
        brand = c["brands"][0]["brand_key"] if c.get("brands") else "-"
        print(f"{c['decision']:8s} {brand:12s} {p[:70]!r}  | {c.get('reason', '')[:90]}")


def cmd_run(args, cfg):
    from .loop import Ctx, run
    if args.repair:
        cfg["edit"]["backend"] = "none" if args.repair == "regen" else cfg["edit"]["backend"]
        if args.repair == "edit" and cfg["edit"]["backend"] == "none":
            cfg["edit"]["backend"] = "comfy"
    if args.editor:
        cfg["edit"]["backend"] = {"qwen": "comfy", "gemini": "openrouter"}[args.editor]
        if args.editor == "gemini":
            cfg["models"]["editor"] = ["google/gemini-3.1-flash-image"]
    if args.generator:
        if args.generator == "flux":
            cfg["generation"]["backend"] = "comfy"
        else:
            cfg["generation"]["backend"] = "openrouter"
            cfg["generation"]["openrouter_image_model"] = "google/gemini-3.1-flash-image"
    if args.mode:
        cfg["generation"]["mode"] = args.mode
    if args.visibility:
        cfg["prominence"]["visibility"] = args.visibility
    if args.max_attempts:
        cfg["loop"]["max_attempts"] = args.max_attempts
    if args.rewriter:
        cfg["rewriter"] = {"v1": {"prompt": "rewriter", "max_inserted_words": 45},
                           "v2": {"prompt": "rewriter_v2", "max_inserted_words": 18},
                           "v3": {"prompt": "rewriter_v3", "max_inserted_words": 12, "max_other_words": 36,
                                  "baseline_aware": True}}[args.rewriter]
    if args.rewriter_model:
        cfg["models"]["rewriter"] = [args.rewriter_model]
        cfg["models"]["repair"] = [args.rewriter_model]
    run_id = args.name or time.strftime("%Y%m%d-%H%M%S")
    llm = make_llm(cfg, args.mock, run_id)
    models = resolve_models(cfg, llm)
    gen, ed = (None, None) if args.dry_run else make_backends(cfg, llm, args.mock)
    if cfg["generation"].get("mode") == "vpp" and ed is None and not args.dry_run:
        sys.exit("--mode vpp needs an edit backend")
    run_dir = ROOT / "runs" / run_id
    ctx = Ctx(cfg=cfg, llm=llm, models=models, gen=gen, editor=ed, catalog=load_catalog(),
              run_dir=run_dir, dry_run=args.dry_run,
              reuse=(ROOT / "runs" / args.reuse) if args.reuse else None, reuse_upto=args.reuse_upto,
              brand_prefs=json.loads(Path(args.brand_prefs).read_text()) if args.brand_prefs else {},
              resume=args.resume)
    prompts = read_prompts(args.file, args.limit)
    print(f"run {run_id}: {len(prompts)} prompts, repair={'regen' if cfg['edit']['backend'] == 'none' else 'edit:' + cfg['edit']['backend']}, gen={cfg['generation']['backend']}, "
          f"mode={cfg['generation'].get('mode', 'rewrite')}, rewriter={(cfg.get('rewriter') or {}).get('prompt')}, models={models}")

    def done(r):
        att = len(r.get("attempts", []))
        print(f"  [{r['idx']:03d}] {r['status']:14s} {r.get('brand') or '':10s} attempts={att} "
              f"${r.get('cost_usd', 0):.4f}  {r['prompt'][:60]!r}"
              + (f"  ERROR {r['error'][:120]}" if r.get("error") else ""), flush=True)

    summary = run(ctx, prompts, workers=args.workers, on_done=done)
    print(json.dumps(summary, indent=1))
    if not args.dry_run:
        from .report import build
        print("report:", build(run_dir))


def cmd_report(args, cfg):
    from .report import build
    print(build(Path(args.run_dir)))


def main(argv=None):
    load_env(ROOT / ".env")
    ap = argparse.ArgumentParser(prog="adplace")
    ap.add_argument("--config")
    ap.add_argument("--mock", action="store_true", help="offline: MockLLM + MockImage")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("models"); s.add_argument("--filter")
    sub.add_parser("check")
    s = sub.add_parser("classify"); s.add_argument("file"); s.add_argument("--limit", type=int)
    s = sub.add_parser("run"); s.add_argument("file"); s.add_argument("--limit", type=int)
    s.add_argument("--dry-run", action="store_true", help="stop after the rewrite (no images)")
    s.add_argument("--repair", choices=["edit", "regen"], help="edit = Qwen-Image-Edit repairs; regen = rewrite+regenerate only")
    s.add_argument("--mode", choices=["rewrite", "vpp"], help="vpp = edit the product into the unmodified baseline")
    s.add_argument("--visibility", choices=["natural", "calibrated", "subtle", "moderate"])
    s.add_argument("--rewriter-model", help="OpenRouter id for the rewriter AND the repair writer (e.g. a closed model); verifier unchanged")
    s.add_argument("--editor", choices=["qwen", "gemini"], help="qwen = Qwen-Image-Edit on Comfy (open); gemini = Gemini 3.1 Flash Image on OpenRouter (closed)")
    s.add_argument("--generator", choices=["flux", "gemini"], help="flux = FLUX.2 dev on Comfy (open); gemini = Gemini 3.1 Flash Image (closed, upper-bound comparison)")
    s.add_argument("--max-attempts", type=int)
    s.add_argument("--brand-prefs", help="JSON {prompt: brand_key}: prefer this brand when the classifier ranked it (brand diversity)")
    s.add_argument("--resume", action="store_true", help="re-run only errored/unfinished prompts of an existing run --name")
    s.add_argument("--rewriter", choices=["v1", "v2", "v3"], help="v1 = detailed spatial clause, v2 = short backgrounded clause, v3 = baseline-aware (sees the same-seed baseline)")
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--name", help="run id (default: timestamp)")
    s.add_argument("--reuse-upto", choices=["attempt1", "plan"], default="attempt1",
                   help="with --reuse: copy through attempt 1 (paired repair comparison) or only through the plan")
    s.add_argument("--reuse", help="earlier run id: reuse its classify/plan/rewrite/baseline/attempt 1 (paired comparison)")
    s = sub.add_parser("report"); s.add_argument("run_dir")
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config)
    {"models": cmd_models, "check": cmd_check, "classify": cmd_classify,
     "run": cmd_run, "report": cmd_report}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
