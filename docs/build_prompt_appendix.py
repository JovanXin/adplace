"""Collect every prompt the pipeline uses into one appendix.

Part A: the LLM stage prompts (system prompts in adplace/prompts/ + the inline ones), with the model,
         temperature and what the user message contains.
Part B: every prompt actually sent to an image model (FLUX.2 dev, plus the edit/closed-model runs),
         per run, prompt and attempt, from the run records.

    .venv/bin/python docs/build_prompt_appendix.py
Writes docs/prompts_appendix.md (for the thesis) and docs/prompts_appendix.html (to publish).
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

P = ROOT / "adplace" / "prompts"
STAGES = [
    ("classifier", "Classifier: INJECT / SKIP / EXCLUDE + brand", "Gemini 3.7 Flash, temp 0.2",
     "JSON: {user_prompt, catalogue} (catalogue = brands.yaml: display, category, fits, avoid, portrayal_rules, restricted, render_reliability). A keyword gate runs first and can EXCLUDE without an LLM call."),
    ("planner", "Planner: scene analysis + 3 candidate placements", "Gemini 3.7 Flash, temp 0.2",
     "JSON: {user_prompt, brand_key, brand (full catalogue entry), visibility_target, classification}. Candidates are then re-ranked deterministically (pipeline.WEIGHTS)."),
    ("rewriter", "Rewriter v1 (pilot A/B): detailed spatial clause", "Gemini 3.7 Flash, temp 0.3 (lint fixes 0.2)",
     "JSON: {original_prompt, brand, plan (chosen candidate), scene, [previous_rewrite, revision_feedback]}"),
    ("rewriter_v2", "Rewriter v2 (pilot G): short, backgrounded clause", "Gemini 3.7 Flash, temp 0.3",
     "Same payload as v1. Lint: product phrase ≤ 18 words, other additions ≤ 20."),
    ("rewriter_v3", "Rewriter v3 (session 2): baseline-aware", "Gemini 3.7 Flash, temp 0.3, with the same-seed baseline image attached",
     "Same payload as v1 + the baseline image. Lint: product phrase ≤ 12 words, other additions ≤ 36."),
    ("repair", "Repair writer: edit instruction or rewriter feedback", "Gemini 3.7 Flash, temp 0.2, failed image (+ baseline under v3) attached",
     "JSON: {action, rewriter, original_prompt, prompt_sent, brand, plan, issues, fix_targets, metrics, inspector_fix_suggestion, target_band}"),
    ("verify_blind", "Verifier, blind pass (naive viewer)", "Gemini 3.7 Flash, temp 0, image attached",
     "Text: Prompt the person typed: '<original prompt>'"),
    ("verify_informed", "Verifier, informed pass (inspector)", "Gemini 3.7 Flash, temp 0, [baseline,] image attached",
     "JSON: {original_prompt, prompt_sent_to_image_model, intended_product {brand, form, anchor, spatial_relation, frame_region, orientation}, [images]}"),
    ("vpp_insert", "VPP condition: choose real estate in the baseline", "Gemini 3.7 Flash, temp 0.2, baseline attached",
     "JSON: {original_prompt, brand, forms, text_plan_hint, target_band}"),
    ("probe_measure", "FLUX probe: measurement against the baseline", "Gemini 3.7 Flash, temp 0, baseline + variant attached",
     "JSON: {original_prompt, modified_prompt, object_class, brand, location_asked}"),
    ("instruction_check", "FLUX instruction test: atomic-requirement check", "Gemini 3.7 Flash, temp 0, baseline + variant attached",
     "JSON: {original_prompt, instruction_appended, full_prompt_sent}"),
]


def inline_prompts() -> list[tuple[str, str, str, str]]:
    from flux_probe import INVENTORY_SYS
    return [("probe_inventory", "FLUX probe: baseline prop inventory (inline in scripts/flux_probe.py)",
             "Gemini 3.7 Flash, temp 0, baseline attached", "Text: Prompt: '<original prompt>'", INVENTORY_SYS)]


def image_prompts() -> list[dict]:
    """[{run, title, rows: [{idx, prompt, attempt, action, text}]}]"""
    runs = []
    order = ["pilot-edit", "pilot-regen", "pilot-v2", "pilot-v2-regen", "pilot-v3",
             "pilot-vpp", "pilot-vpp-gemini", "pilot-gemini-gen"]
    for name in order:
        d = ROOT / "runs" / name
        if not d.exists():
            continue
        cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
        gen = cfg.get("generation", {})
        model = "FLUX.2 dev (Comfy)" if gen.get("backend") == "comfy" else gen.get("openrouter_image_model", "?")
        editor = {"comfy": "Qwen-Image-Edit-2511", "openrouter": "Gemini image", "none": "none"}.get(
            cfg.get("edit", {}).get("backend"), "?")
        rows = []
        for rp in sorted(d.glob("[0-9][0-9][0-9]/record.json")):
            r = json.loads(rp.read_text())
            if not r.get("attempts"):
                continue
            rows.append({"idx": r["idx"], "prompt": r["prompt"], "attempt": 0, "action": "baseline (original prompt)",
                         "text": r["prompt"]})
            for a in r["attempts"]:
                mb = a.get("made_by") or {}
                if mb.get("action") == "edit":
                    txt = " ⟶ ".join(mb.get("steps") or [mb.get("instruction", "")])
                    act = f"edit ({mb.get('mode', '')}) · {editor}"
                else:
                    txt = a.get("prompt") or mb.get("prompt", "")
                    act = f"generate ({mb.get('mode') or 'first'}) · seed {mb.get('seed', '')}" + (" · reused" if mb.get("reused") else "")
                rows.append({"idx": r["idx"], "prompt": r["prompt"], "attempt": a["k"], "action": act, "text": txt})
        runs.append({"run": name, "title": f"{name} · generator {model} · edit {editor} · "
                     f"rewriter {(cfg.get('rewriter') or {}).get('prompt', 'rewriter')}", "rows": rows})
    # probe
    pj = ROOT / "runs" / "flux-probe" / "probe.json"
    if pj.exists():
        from flux_probe import SCENES
        db = json.loads(pj.read_text())
        rows = []
        for key, e in db.items():
            rows.append({"idx": key, "prompt": SCENES[e["scene"]]["orig"], "attempt": 0, "action": "B0 baseline",
                         "text": SCENES[e["scene"]]["orig"]})
            for c, ce in e["conds"].items():
                rows.append({"idx": key, "prompt": SCENES[e["scene"]]["orig"], "attempt": c, "action": c, "text": ce["prompt"]})
        runs.append({"run": "flux-probe", "title": "flux-probe · FLUX.2 dev · 3 scenes × 4 seeds × 8 phrasings", "rows": rows})
    ij = ROOT / "runs" / "flux-instructions" / "results.json"
    if ij.exists():
        from flux_instructions import INSTRUCTIONS, SCENES as IS
        rows = [{"idx": iid, "prompt": IS[sc], "attempt": cat, "action": f"{cat} · seeds 5491897, 1234, 777, 42",
                 "text": f"{IS[sc]}, {instr}"} for iid, sc, cat, instr in INSTRUCTIONS]
        runs.append({"run": "flux-instructions", "title": "flux-instructions · FLUX.2 dev · 16 instructions × 4 seeds", "rows": rows})
    return runs


def build():
    stages = [(k, t, m, u, (P / f"{k}.md").read_text()) for k, t, m, u in STAGES] + inline_prompts()
    runs = image_prompts()

    # ---------- markdown ----------
    md = ["# Appendix: prompts used by adplace", "",
          "Generated by `docs/build_prompt_appendix.py` from `adplace/prompts/` and the run records in `runs/`.", "",
          "## A. LLM stage prompts", ""]
    for k, t, m, u, body in stages:
        md += [f"### A.{stages.index((k, t, m, u, body)) + 1} {t}", "", f"- File: `{k}`", f"- Model: {m}",
               f"- User message: {u}", "", "```text", body.rstrip(), "```", ""]
    md += ["## B. Prompts sent to the image model", ""]
    for i, run in enumerate(runs, 1):
        md += [f"### B.{i} {run['title']}", "", "| # | step | prompt / instruction |", "|---|---|---|"]
        for r in run["rows"]:
            md.append(f"| {r['idx']} | {r['attempt']} · {r['action']} | {r['text'].replace('|', '\\|')} |")
        md.append("")
    (ROOT / "docs" / "prompts_appendix.md").write_text("\n".join(md))

    # ---------- html ----------
    esc = html.escape
    nav = "".join(f'<a href="#{k}">{esc(t.split(":")[0])}</a>' for k, t, *_ in stages)
    navb = "".join(f'<a href="#run-{r["run"]}">{esc(r["run"])}</a>' for r in runs)
    secA = "".join(f"""
    <section class="stage" id="{k}">
      <div class="head"><h3>{esc(t)}</h3><button class="copy" data-for="pre-{k}">Copy</button></div>
      <dl><dt>File</dt><dd><code>adplace/prompts/{k}.md</code></dd><dt>Model</dt><dd>{esc(m)}</dd><dt>User message</dt><dd>{esc(u)}</dd></dl>
      <pre id="pre-{k}">{esc(body.rstrip())}</pre>
    </section>""" for k, t, m, u, body in stages)
    secB = ""
    for r in runs:
        body = "".join(
            f"<tr><td class='mono'>{esc(str(x['idx']))}</td><td class='mono'>{esc(str(x['attempt']))}</td>"
            f"<td class='act'>{esc(x['action'])}</td><td>{esc(x['text'])}</td></tr>" for x in r["rows"])
        secB += f"""
    <details class="run" id="run-{r['run']}"><summary><b>{esc(r['run'])}</b> <span>{esc(r['title'].split(' · ', 1)[1] if ' · ' in r['title'] else '')} · {len(r['rows'])} prompts</span></summary>
      <div class="tbl"><table><thead><tr><th>#</th><th>Step</th><th>How</th><th>Prompt or edit instruction</th></tr></thead><tbody>{body}</tbody></table></div>
    </details>"""
    tpl = (ROOT / "docs" / "prompts_appendix_template.html").read_text()
    out = tpl.replace("<!--NAV_A-->", nav).replace("<!--NAV_B-->", navb).replace("<!--SEC_A-->", secA).replace("<!--SEC_B-->", secB)
    out = out.replace("<!--COUNTS-->", f"{len(stages)} LLM prompts · {sum(len(r['rows']) for r in runs)} image-model prompts across {len(runs)} runs")
    (ROOT / "docs" / "prompts_appendix.html").write_text(out)
    print("wrote docs/prompts_appendix.md and .html:", len(stages), "stages,", sum(len(r["rows"]) for r in runs), "image prompts")


if __name__ == "__main__":
    build()
