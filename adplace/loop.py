"""Orchestrator: classify -> plan -> rewrite -> generate -> verify -> repair (edit/regenerate) -> ...

Repair policy (one action per failed attempt, driven by the most severe issue):
  FIDELITY_LOSS   regenerate with the next-best plan candidate (an edit can't restore composition)
  ARTIFACTS       regenerate the same prompt with a new seed
  ABSENT          regenerate with a stronger cue; after 2 ABSENT verdicts, edit the product in
  ON_SUBJECT / DUPLICATES / TOO_PROMINENT / GARBLED_BRAND / BAD_INTEGRATION
                  edit (cheap, keeps the scene); if an edit already failed on that code, regenerate
                  with rewriter feedback instead
The concrete edit instruction / rewriter feedback is written by a vision LLM (prompts/repair.md)
that sees the failed image; a template is used if that call fails.
"""
from __future__ import annotations

import csv
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .llm import BudgetExceeded
from .pipeline import classify, load_prompt, plan, rewrite
from .verify import PRIORITY, informed_pass, severity, verify

def _rw_opts(cfg: dict) -> dict:
    r = cfg.get("rewriter") or {}
    return {"prompt_name": r.get("prompt", "rewriter"), "max_words": int(r.get("max_inserted_words", 45)),
            "max_other": int(r.get("max_other_words", 20))}


EDITABLE = {"ON_SUBJECT", "DUPLICATES", "TOO_PROMINENT", "GARBLED_BRAND", "BAD_INTEGRATION"}
KEEP = "Keep everything else in the image exactly the same."


@dataclass
class Ctx:
    cfg: dict
    llm: object
    models: dict            # classifier / rewriter / verifier -> resolved model ids
    gen: object             # .generate(prompt, seed, w, h) -> bytes
    editor: object | None   # .edit(image, instruction, seed) -> bytes, or None
    catalog: dict
    run_dir: Path
    dry_run: bool = False
    reuse: Path | None = None   # earlier run: reuse its classify/plan/rewrite/baseline/attempt 1 (paired design)
    reuse_upto: str = "attempt1"  # attempt1 | plan (re-run the rewrite and everything after it)
    resume: bool = False          # keep finished prompts already in run_dir; redo errored/unfinished ones
    brand_prefs: dict = field(default_factory=dict)  # prompt -> preferred brand key (used only if the classifier ranked it)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def band(self) -> dict:
        p = self.cfg["prominence"]
        return p[p["visibility"]]


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------
def choose_action(issues: list[dict], history: list[dict], can_edit: bool) -> dict:
    codes = [i["code"] for i in issues]
    primary = codes[0]
    n_absent = 1 + sum(1 for h in history if "ABSENT" in h.get("codes", []))  # incl. this verdict
    failed_edits = {c for h in history if h.get("action") == "edit"
                    for c in h.get("targets", []) if c in codes}

    if primary == "FIDELITY_LOSS":
        return {"action": "regenerate", "mode": "replan", "targets": [primary]}
    if primary == "ARTIFACTS":
        return {"action": "regenerate", "mode": "reseed", "targets": [primary]}
    if primary == "ABSENT":
        if n_absent >= 2 and can_edit:
            return {"action": "edit", "mode": "insert", "targets": ["ABSENT"]}
        return {"action": "regenerate", "mode": "feedback", "targets": ["ABSENT"]}
    targets = [c for c in codes if c in EDITABLE]
    n_feedback = sum(1 for h in history if h.get("mode") == "feedback" and primary in h.get("targets", []))
    if n_feedback >= 2:     # two prompt revisions did not fix it: try a different placement candidate
        return {"action": "regenerate", "mode": "replan", "targets": [primary]}
    if can_edit and not (set(targets) & failed_edits):
        mode = "remove_reinsert" if "TOO_PROMINENT" in targets else "fix"
        return {"action": "edit", "mode": mode, "targets": targets}
    return {"action": "regenerate", "mode": "feedback", "targets": targets or [primary]}


def _template(action: dict, brand: dict, chosen: dict, issues: list[dict]) -> dict:
    form, rel = chosen.get("form", "product"), chosen.get("spatial_relation", "in the background")
    name = f"{brand['display']} {form}"
    parts = []
    for code in action["targets"]:
        if code == "ABSENT":
            parts.append(f"Add one {name} {rel}, {chosen.get('scale_hint', 'small')}, "
                         f"{chosen.get('orientation', 'label turned partly away')}, "
                         f"{chosen.get('lighting', 'matching the scene lighting')}.")
        elif code == "TOO_PROMINENT":
            # Qwen-Image-Edit rescales poorly but removes/adds well: remove, then re-insert small.
            return {"edit_steps": [f"Remove the {name} and fill the area with the surrounding scene. {KEEP}",
                                   f"Add one small {name} {rel}, {chosen.get('scale_hint', 'small')}, "
                                   f"label turned away, in the background. {KEEP}"],
                    "edit_instruction": "", "revision_feedback":
                        f"Make the {form} much smaller and further back {rel}, label turned away.",
                    "rationale": "template"}
        elif code == "DUPLICATES":
            parts.append(f"Remove every {form} except the single {name} {rel}.")
        elif code == "ON_SUBJECT":
            parts.append(f"Move the {name} to {rel}.")
        elif code == "GARBLED_BRAND":
            parts.append(f"Correct the text on the {form} so it reads '{brand['display']}'.")
        elif code == "BAD_INTEGRATION":
            parts.append(f"Make the {name} match the scene's lighting, perspective and style, "
                         f"resting on its surface with a soft contact shadow.")
    fb = " ".join(parts) or "Make the placement smaller and further from the main subject."
    return {"edit_instruction": f"{fb} {KEEP}", "edit_steps": [f"{fb} {KEEP}"], "revision_feedback": fb,
            "rationale": "template"}


def repair_text(ctx: Ctx, image: bytes, action: dict, *, original: str, rewritten: str,
                brand: dict, chosen: dict, v: dict, baseline: bytes | None = None) -> dict:
    if not ctx.cfg["loop"].get("llm_repair", True):
        return _template(action, brand, chosen, v["issues"])
    payload = {"action": action["action"], "rewriter": (ctx.cfg.get("rewriter") or {}).get("prompt"),
               "original_prompt": original, "prompt_sent": rewritten,
               "brand": brand["display"], "plan": chosen, "issues": v["issues"],
               "fix_targets": action["targets"], "metrics": v["metrics"],
               "inspector_fix_suggestion": v["informed"].get("fix_suggestion"),
               "target_band": ctx.band}
    try:
        if baseline is not None:
            payload["images"] = "FIRST = the failed image; SECOND = the same-seed baseline (original prompt)"
        out = ctx.llm.chat_json(ctx.models.get("repair") or ctx.models["verifier"], load_prompt("repair"), payload,
                                task="repair", temperature=0.2,
                                images=[image, baseline] if baseline is not None else [image])
        if action["action"] == "edit":
            steps = [x for x in (out.get("edit_steps") or [out.get("edit_instruction", "")]) if str(x).strip()]
            if not steps:
                raise ValueError("no edit instruction")
            out["edit_steps"] = [x if KEEP.lower() in x.lower() else f"{x.rstrip()} {KEEP}" for x in steps[:2]]
        elif not str(out.get("revision_feedback", "")).strip():
            raise ValueError("empty revision_feedback")
        return out
    except BudgetExceeded:
        raise
    except Exception as e:  # noqa: BLE001 - any LLM failure falls back to the template
        t = _template(action, brand, chosen, v["issues"])
        t["rationale"] = f"template (repair LLM failed: {e})"
        return t


def vpp_plan(ctx: Ctx, image: bytes, prompt: str, brand: dict, chosen: dict) -> dict | None:
    payload = {"original_prompt": prompt, "brand": brand["display"],
               "forms": brand.get("forms"), "text_plan_hint": chosen, "target_band": ctx.cfg["prominence"]["visibility"]}
    try:
        out = ctx.llm.chat_json(ctx.models["verifier"], load_prompt("vpp_insert"), payload,
                                task="vpp_plan", temperature=0.2, images=[image])
        if not str(out.get("instruction", "")).strip():
            return None
        if brand["display"].lower() not in out["instruction"].lower():   # an unbranded insert is useless
            form = out.get("form") or chosen.get("form", "product")
            out["instruction"] = (f"Add one small {brand['display']} {form} {out.get('anchor') or 'in the background'}, "
                                  f"partly turned away, matching the scene's style and lighting.")
            out["rationale"] = (out.get("rationale") or "") + " [brand missing: rebuilt from anchor]"
        if KEEP.lower() not in out["instruction"].lower():
            out["instruction"] = f"{out['instruction'].rstrip()} {KEEP}"
        return out
    except BudgetExceeded:
        raise
    except Exception:  # noqa: BLE001 - fall back to the text-plan instruction
        return None


def vpp_instruction(brand: dict, chosen: dict, medium: str = "") -> str:
    """Insert instruction for the VPP condition, from the plan (short: the editor over-weights long text too)."""
    style = f", in the same {medium} style and lighting" if medium else ", matching the scene's style and lighting"
    return (f"Add one small {brand['display']} {chosen.get('form', 'product')} "
            f"{chosen.get('spatial_relation', 'in the background')}, partly turned away{style}. {KEEP}")


# --------------------------------------------------------------------------
# One prompt
# --------------------------------------------------------------------------
def _save(path: Path, rec: dict):
    path.write_text(json.dumps(rec, indent=1, default=str))


def process_prompt(ctx: Ctx, idx: int, prompt: str) -> dict:
    cfg = ctx.cfg
    d = ctx.run_dir / f"{idx:03d}"
    d.mkdir(parents=True, exist_ok=True)
    g = cfg["generation"]
    seed, W, H = int(g["seed"]), int(g["width"]), int(g["height"])
    rec: dict = {"idx": idx, "prompt": prompt, "seed": seed, "status": "started",
                 "mode": g.get("mode", "rewrite"), "attempts": [], "t0": time.time()}
    ctx.llm.ledger.begin(f"{ctx.run_dir.name}/{idx:03d}")
    rp = d / "record.json"
    if ctx.resume and rp.exists():
        old = json.loads(rp.read_text())
        if old.get("status") in ("accepted", "failed", "skip", "exclude", "dry_run") and old.get("prompt") == prompt:
            return old

    def finish(status: str):
        rec["status"] = status
        rec["cost_usd"] = round(ctx.llm.ledger.thread_spent(), 5)
        rec["seconds"] = round(time.time() - rec["t0"], 1)
        _save(rp, rec)
        return rec

    prev, prev_dir = None, None
    if ctx.reuse:
        prev_dir = ctx.reuse / f"{idx:03d}"
        if (prev_dir / "record.json").exists():
            prev = json.loads((prev_dir / "record.json").read_text())
            if prev.get("prompt") != prompt:
                prev = None
    try:
        vis = cfg["prominence"]["visibility"]
        if prev:
            rec["reused_from"] = str(ctx.reuse.name)
            c = prev["classify"]
        else:
            c = classify(ctx.llm, ctx.models["classifier"], prompt, ctx.catalog)
        rec["classify"] = c
        if c["decision"] != "INJECT":
            return finish(c["decision"].lower())
        ranked = [b["brand_key"] for b in c["brands"]]
        want = ctx.brand_prefs.get(prompt)
        brand_key = want if want in ranked else ranked[0]
        if want:
            rec["brand_pref"] = {"wanted": want, "ranked": ranked, "used": brand_key}
        brand = ctx.catalog[brand_key]
        rec["brand"] = brand_key
        if prev and prev.get("plan") and prev.get("rewrite") and ctx.reuse_upto == "attempt1":
            p, rw = prev["plan"], prev["rewrite"]
        elif prev and prev.get("plan"):
            p = prev["plan"]
            rw = rewrite(ctx.llm, ctx.models["rewriter"], prompt, brand_key, ctx.catalog, p, **_rw_opts(cfg))
        else:
            p = plan(ctx.llm, ctx.models["rewriter"], prompt, brand_key, ctx.catalog, vis, classification=c)
            rw = rewrite(ctx.llm, ctx.models["rewriter"], prompt, brand_key, ctx.catalog, p, **_rw_opts(cfg))
        rec["plan"], rec["rewrite"] = p, rw
        _save(rp, rec)
        if ctx.dry_run:
            return finish("dry_run")

        baseline = None
        if prev and prev.get("baseline") and (prev_dir / prev["baseline"]).exists():
            baseline = (prev_dir / prev["baseline"]).read_bytes()
            (d / "baseline.png").write_bytes(baseline)
            rec["baseline"] = "baseline.png"
        elif g.get("also_generate_baseline", True) or rec["mode"] == "vpp":
            baseline = ctx.gen.generate(prompt, seed, W, H)
            (d / "baseline.png").write_bytes(baseline)
            rec["baseline"] = "baseline.png"

        # v3: the rewriter looks at the same-seed baseline first. FLUX keeps the baseline's layout when
        # the prompt only grows a little, so anchoring on objects that are really there (and naming the
        # props that must survive) is the prompt-only analogue of VPP picking real estate in the frame.
        see_base = bool((cfg.get("rewriter") or {}).get("baseline_aware")) and baseline is not None
        if see_base and not (prev and (prev.get("attempts") or []) and ctx.reuse_upto == "attempt1"):
            rw = rewrite(ctx.llm, ctx.models["rewriter"], prompt, brand_key, ctx.catalog, p,
                         baseline_image=baseline, **_rw_opts(cfg))
            rec["rewrite"] = rw
        current_prompt = rw.get("rewritten_prompt") or prompt
        chosen = p.get("chosen") or {}
        tried_ids = [chosen.get("id")]
        cur_seed = seed
        can_edit = ctx.editor is not None
        history: list[dict] = []
        pre_edit: tuple[bytes, list | None] | None = None

        prev_att = (prev or {}).get("attempts") or []
        prev_att = prev_att[0] if (prev_att and ctx.reuse_upto == "attempt1"
                                   and rec["mode"] == prev.get("mode", "rewrite")
                                   and (prev_dir / prev_att[0]["image"]).exists()) else None
        if prev_att:
            img = (prev_dir / prev_att["image"]).read_bytes()
            how = {**prev_att["made_by"], "reused": True}
        elif rec["mode"] == "vpp":
            # Virtual-product-placement condition: the image model never sees the brand; the
            # product is edited into the unmodified baseline, like Mirriad/Amazon VPP in post.
            # A vision LLM picks real estate that is actually visible in the baseline.
            vp = vpp_plan(ctx, baseline, prompt, brand, chosen)
            rec["vpp_plan"] = vp
            if vp:
                instr = vp["instruction"]
                chosen = {**chosen, "anchor": vp.get("anchor") or chosen.get("anchor"),
                          "form": vp.get("form") or chosen.get("form"),
                          "spatial_relation": vp.get("anchor") or chosen.get("spatial_relation")}
            else:
                instr = vpp_instruction(brand, chosen, (p.get("scene") or {}).get("medium_style", ""))
            img = ctx.editor.edit(baseline, instr, seed)
            pre_edit = (baseline, None)
            how = {"action": "edit", "mode": "vpp_insert", "instruction": instr}
        else:
            img = ctx.gen.generate(current_prompt, cur_seed, W, H)
            how = {"action": "generate", "prompt": current_prompt, "seed": cur_seed}

        exclude_boxes = None
        if rec["mode"] == "vpp" and baseline is not None:
            # FLUX sometimes draws the brand by itself (the 1950s diner is full of Coca-Cola signs).
            # Those instances are not ours, so they must not count as duplicates or as the placement.
            bl = informed_pass(ctx.llm, ctx.models["verifier"], baseline,
                               {"original_prompt": prompt, "intended_product": {"brand": brand["display"], "form": "any"},
                                "note": "This is the image BEFORE any placement. Box every existing instance of the brand."})
            exclude_boxes = bl.get("product_boxes") if bl.get("product_found") else []
            rec["baseline_brand"] = {"found": bool(bl.get("product_found")), "boxes": exclude_boxes}

        max_attempts = int(cfg["loop"]["max_attempts"])
        best = None
        for k in range(1, max_attempts + 1):
            name = f"attempt_{k}.png"
            (d / name).write_bytes(img)
            v = prev_att["verify"] if (k == 1 and prev_att) else verify(ctx.llm, ctx.models["verifier"], img, original_prompt=prompt,
                       rewritten_prompt=current_prompt, brand=brand, plan_chosen=chosen,
                       band=ctx.band, min_fidelity=cfg["prominence"]["min_fidelity"],
                       min_integration=cfg["prominence"]["min_integration"],
                       pre_edit_image=pre_edit[0] if pre_edit else None,
                       pre_edit_box=pre_edit[1] if pre_edit else None, exclude_boxes=exclude_boxes,
                       baseline_image=baseline if cfg["prominence"].get("compare_baseline") else None,
                       max_scene_loss=float(ctx.band.get("max_scene_loss", cfg["prominence"].get("max_scene_loss", 0.5))))
            att = {"k": k, "image": name, "made_by": how, "prompt": current_prompt,
                   "plan_chosen": chosen, "verify": v,
                   "codes": [i["code"] for i in v["issues"]], "severity": severity(v["issues"])}
            rec["attempts"].append(att)
            if best is None or att["severity"] < best["severity"]:
                best = att
            _save(rp, rec)
            if v["accepted"]:
                rec["final"] = name
                return finish("accepted")
            if k == max_attempts:
                break

            action = choose_action(v["issues"], history, can_edit)
            att["next_action"] = action
            history.append({"codes": att["codes"], **action})

            if action["action"] == "edit":
                txt = repair_text(ctx, img, action, original=prompt, rewritten=current_prompt,
                                  brand=brand, chosen=chosen, v=v)
                att["repair"] = txt
                pre_edit = (img, v["metrics"].get("largest_box"))
                steps = txt.get("edit_steps") or [txt["edit_instruction"]]
                for j, instr in enumerate(steps):
                    img = ctx.editor.edit(img, instr, cur_seed + k)
                    if j < len(steps) - 1:
                        (d / f"attempt_{k + 1}_step{j + 1}.png").write_bytes(img)
                how = {"action": "edit", "mode": action["mode"], "targets": action["targets"],
                       "instruction": " ⟶ ".join(steps), "steps": steps}
                continue

            # ---- regenerate --------------------------------------------------
            pre_edit = None
            if action["mode"] == "reseed":
                cur_seed = seed + 1000 * k
            elif action["mode"] == "replan":
                p2 = plan(ctx.llm, ctx.models["rewriter"], prompt, brand_key, ctx.catalog, vis,
                          avoid_ids=tried_ids, classification=c, previous=p)
                if p2.get("chosen", {}).get("id") in tried_ids:   # candidates exhausted -> fresh plan
                    fails = [{"candidate": a.get("plan_chosen", chosen), "issues": a["codes"]}
                             for a in rec["attempts"]]
                    p2 = plan(ctx.llm, ctx.models["rewriter"], prompt, brand_key, ctx.catalog, vis,
                              classification=c, previous_failures=fails)
                p, chosen = p2, p2.get("chosen") or chosen
                tried_ids.append(chosen.get("id"))
                rw2 = rewrite(ctx.llm, ctx.models["rewriter"], prompt, brand_key, ctx.catalog, p,
                              baseline_image=baseline if see_base else None, **_rw_opts(cfg))
                current_prompt = rw2.get("rewritten_prompt") or current_prompt
                att["replan"] = {"chosen": chosen, "rewrite": rw2}
            else:  # feedback
                txt = repair_text(ctx, img, action, original=prompt, rewritten=current_prompt,
                                  brand=brand, chosen=chosen, v=v, baseline=baseline if see_base else None)
                att["repair"] = txt
                rw2 = rewrite(ctx.llm, ctx.models["rewriter"], prompt, brand_key, ctx.catalog, p,
                              previous=current_prompt, feedback=txt["revision_feedback"],
                              baseline_image=baseline if see_base else None, **_rw_opts(cfg))
                current_prompt = rw2.get("rewritten_prompt") or current_prompt
                att["rewrite_after"] = rw2
            img = ctx.gen.generate(current_prompt, cur_seed, W, H)
            how = {"action": "generate", "mode": action["mode"], "prompt": current_prompt,
                   "seed": cur_seed, "targets": action["targets"]}

        rec["final"] = best["image"] if best else None
        return finish("failed")
    except BudgetExceeded as e:
        rec["error"] = str(e)
        return finish("budget_stopped")
    except Exception as e:  # noqa: BLE001 - one bad prompt must not kill the run
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()
        return finish("error")


# --------------------------------------------------------------------------
# A run
# --------------------------------------------------------------------------
def run(ctx: Ctx, prompts: list[str], workers: int = 1, on_done=None) -> dict:
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    (ctx.run_dir / "config.json").write_text(json.dumps(ctx.cfg, indent=1))
    (ctx.run_dir / "models.json").write_text(json.dumps(ctx.models, indent=1))
    t0, s0 = time.time(), ctx.llm.ledger.spent

    def one(ip):
        r = process_prompt(ctx, *ip)
        if on_done:
            on_done(r)
        return r

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        recs = list(ex.map(one, list(enumerate(prompts))))
    summary = summarise(recs)
    summary.update({"seconds": round(time.time() - t0, 1),
                    "openrouter_usd": round(ctx.llm.ledger.spent - s0, 4),
                    "images_generated": ctx.llm.ledger.images})
    (ctx.run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    write_disclosure_log(ctx.run_dir, recs)
    return summary


def summarise(recs: list[dict]) -> dict:
    from collections import Counter
    status = Counter(r["status"] for r in recs)
    injected = [r for r in recs if r.get("attempts")]
    codes = Counter(c for r in injected for a in r["attempts"] for c in a["codes"])
    first_pass = sum(1 for r in injected if r["attempts"][0]["verify"]["accepted"])
    actions = Counter(a.get("next_action", {}).get("action") for r in injected
                      for a in r["attempts"] if a.get("next_action"))
    return {
        "n_prompts": len(recs), "status": dict(status),
        "decisions": dict(Counter((r.get("classify") or {}).get("decision", "?") for r in recs)),
        "n_generated": len(injected),
        "accept_rate": round(status.get("accepted", 0) / len(injected), 3) if injected else None,
        "first_pass_accept_rate": round(first_pass / len(injected), 3) if injected else None,
        "mean_attempts": round(sum(len(r["attempts"]) for r in injected) / len(injected), 2) if injected else None,
        "issue_codes": {c: codes.get(c, 0) for c in PRIORITY},
        "repair_actions": {k: v for k, v in actions.items() if k},
        "blind_brand_recognised": sum(1 for r in injected if _final_att(r) and _final_att(r)["verify"]["brand_seen_blind"]),
    }


def _final_att(r: dict) -> dict | None:
    return next((a for a in r.get("attempts", []) if a["image"] == r.get("final")), None)


def write_disclosure_log(run_dir: Path, recs: list[dict]):
    """Every injection, for post-study disclosure (Thesis A ethics requirement)."""
    with (run_dir / "injections.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "status", "brand", "original_prompt", "final_prompt", "final_image"])
        for r in recs:
            if r.get("brand"):
                fa = _final_att(r) or {}
                fp = fa.get("prompt") or (r.get("rewrite") or {}).get("rewritten_prompt")
                if (fa.get("made_by") or {}).get("action") == "edit":
                    fp = f"{fp}  [then edited: {fa['made_by'].get('instruction')}]"
                w.writerow([r["idx"], r["status"], r["brand"], r["prompt"], fp,
                            f"{r['idx']:03d}/{r.get('final')}" if r.get("final") else ""])
