"""Session 3: survey image set on closed image models + rewriter / feedback-loop benchmark.

Per prompt (all images on one seed with gemini-3.1-flash-lite-image, 1:1):
  base   original prompt                                      -> survey "no-inject"
  bad    product named first, in the foreground (blatant)     -> survey "bad-inject"
  naive  original + ", a <Brand> <form>"           (no LLM)
  v2     rewriter_v2 (FLUX-era short backgrounded phrase), Gemini 3.7 Flash
  v4f    rewriter_v4 (closed-model aware, sees base), Gemini 3.7 Flash
  v4s    rewriter_v4, Claude Sonnet 5.5 (closed LLM)
  loop   best v4 variant + repair loop (vision feedback -> revised prompt -> regenerate), <= 2 extra tries
Every injected image: Gemini 3.7 Flash verifier (blind + informed), judged with the `calibrated` band
(the rule fitted to Jovan's lightbox labels). Every image: GPT-5.6 Sol survey-proxy judge + CLIPScore
against the ORIGINAL prompt. Unpaired caveat: closed models re-draw the whole scene when the prompt
changes, even on a fixed seed (scripts/closed_seed_test.py).

    .venv/bin/python scripts/survey_bench.py gen|verify|judge|clip|loop|select [--only 0,3]
State: runs/survey-closed/<idx>/state.json (resumable; every API result cached).
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
from adplace.__main__ import load_cfg, make_llm  # noqa: E402
from adplace.llm import load_env  # noqa: E402
from adplace.pipeline import classify, load_catalog, load_prompt, plan, rewrite  # noqa: E402
from adplace.verify import verify  # noqa: E402

OUT = ROOT / "runs" / "survey-closed"
IMG_MODEL = "google/gemini-3.1-flash-lite-image"
FLASH = "google/gemini-3.7-flash"
SONNET = "anthropic/claude-sonnet-5.5"
JUDGE = "openai/gpt-5.6-sol"
SEED = 7
V2_LIMIT = 12                      # v2 only on the first 12 prompts (reference condition, budget)
COND_ORDER = ["base", "bad", "naive", "v2", "v4f", "v4s"]
_locks: dict[int, threading.Lock] = {}

load_env(ROOT / ".env")
CFG = load_cfg(None)
LLM = make_llm(CFG, False, run_id="survey-closed")
CAT = load_catalog()
PREFS = json.loads((ROOT / "survey_brand_prefs.json").read_text())
PROMPTS = [l.strip() for l in (ROOT / "survey_prompts.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
BAND = CFG["prominence"]["calibrated"]


def sdir(i): return OUT / f"{i:03d}"


def load(i):
    p = sdir(i) / "state.json"
    return json.loads(p.read_text()) if p.exists() else {"idx": i, "prompt": PROMPTS[i], "cond": {}}


def save(i, st):
    sdir(i).mkdir(parents=True, exist_ok=True)
    (sdir(i) / "state.json").write_text(json.dumps(st, indent=1))


def lock(i):
    return _locks.setdefault(i, threading.Lock())


def short_form(form: str) -> str:
    return form.split(" with ")[0].split(",")[0].strip()


def gen_image(prompt: str) -> bytes:
    return LLM.image(IMG_MODEL, prompt, task="survey_gen", seed=SEED, width=1024, height=1024)


# ------------------------------------------------------------------ gen
def do_gen(i):
    st = load(i)
    p = st["prompt"]
    if "classify" not in st:
        c = classify(LLM, FLASH, p, CAT)
        ranked = [b["brand_key"] for b in c.get("brands", [])]
        if c["decision"] != "INJECT" or not ranked:
            st.update(classify=c, skip=True); save(i, st); return
        want = PREFS.get(p)
        st.update(classify=c, brand=want if want in ranked else ranked[0])
        st["plan"] = plan(LLM, FLASH, p, st["brand"], CAT, "natural", classification=c)
        save(i, st)
    b = CAT[st["brand"]]
    prod = f'{b["display"]} {short_form((st["plan"].get("chosen") or {}).get("form") or b["forms"][0]["form"])}'
    st["product"] = prod
    conds = st["cond"]
    if "base" not in conds:
        (sdir(i) / "base.png").write_bytes(gen_image(p)); conds["base"] = {"prompt": p}; save(i, st)
    base = (sdir(i) / "base.png").read_bytes()
    art = "An" if prod[0].lower() in "aeiou" else "A"
    todo = {"bad": lambda: {"prompt": f"{art} {prod} in the foreground, {p}"},
            "naive": lambda: {"prompt": f"{p}, a {prod}"}}
    if i < V2_LIMIT:
        todo["v2"] = lambda: rw("rewriter_v2", FLASH, None, 18)
    todo["v4f"] = lambda: rw("rewriter_v4", FLASH, base, 18)
    todo["v4s"] = lambda: rw("rewriter_v4", SONNET, base, 18)

    def rw(name, model, img, mw):
        out = rewrite(LLM, model, p, st["brand"], CAT, st["plan"], prompt_name=name, max_words=mw,
                      max_other=24, baseline_image=img)
        return {"prompt": out.get("rewritten_prompt") or p, "rewrite": out}

    for k, f in todo.items():
        if k in conds and (sdir(i) / f"{k}.png").exists():
            continue
        c = f()
        (sdir(i) / f"{k}.png").write_bytes(gen_image(c["prompt"]))
        conds[k] = c
        save(i, st)


# ------------------------------------------------------------------ verify / judge / clip
def verify_one(st, key, img_path, sent_prompt):
    b = CAT[st["brand"]]
    pr = CFG["prominence"]
    return verify(LLM, FLASH, img_path.read_bytes(), original_prompt=st["prompt"], rewritten_prompt=sent_prompt,
                  brand=b, plan_chosen=st["plan"].get("chosen"), band=BAND, min_fidelity=pr["min_fidelity"],
                  min_integration=pr["min_integration"], baseline_image=None,
                  max_scene_loss=float(BAND.get("max_scene_loss", 0.75)))


def judge_one(prompt, img_path):
    return LLM.chat_json(JUDGE, load_prompt("survey_judge"), f"Prompt the person typed: {prompt!r}",
                         task="survey_judge", temperature=0.0, images=[img_path.read_bytes()])


def items(st):
    for k, c in st["cond"].items():
        yield k, c
    for j, a in enumerate(st.get("loop", []), 1):
        yield f"loop{j}", a


def do_verify(i):
    st = load(i)
    if st.get("skip"): return
    for k, c in items(st):
        if k == "base" or c.get("verify"): continue
        c["verify"] = verify_one(st, k, sdir(i) / f"{k}.png", c["prompt"])
        with lock(i): save(i, st)


def do_judge(i):
    st = load(i)
    if st.get("skip"): return
    for k, c in items(st):
        if c.get("judge"): continue
        try:
            c["judge"] = judge_one(st["prompt"], sdir(i) / f"{k}.png")
        except Exception as e:  # noqa: BLE001
            c["judge_error"] = str(e)[:200]
        with lock(i): save(i, st)


def do_clip(i):
    from adplace.clipscore import clip_cos
    st = load(i)
    if st.get("skip"): return
    for k, c in items(st):
        if "clip" not in c:
            c["clip"] = clip_cos(sdir(i) / f"{k}.png", st["prompt"])
    save(i, st)


def brand_seen(st, j) -> bool:
    key = CAT[st["brand"]]["display"].lower().replace("-", "").replace(" ", "")
    return any(key in str(b).lower().replace("-", "").replace(" ", "") for b in (j or {}).get("brands", []))


def survey_good(st, c) -> bool:
    """Good-inject for the survey: passes Jovan's calibrated rule, a fresh viewer (GPT judge) can name the
    brand, yet rates it at most 3/5 'looks like an ad', and the image still does what the prompt asked."""
    v, j = c.get("verify") or {}, c.get("judge") or {}
    return bool(v.get("accepted")) and brand_seen(st, j) and (j.get("ad_likelihood") or 5) <= 3 \
        and (j.get("adherence") or 0) >= 4


def survey_bad(c) -> bool:
    v, j = c.get("verify") or {}, c.get("judge") or {}
    return not v.get("accepted", True) and (j.get("ad_likelihood") or 0) >= 4


# ------------------------------------------------------------------ rerolls
def do_reroll_bad(i, tries: int = 2):
    st = load(i)
    if survey_bad(st["cond"]["bad"]) or st.get("bad_reroll"): return
    prod, p = st["product"], st["prompt"]
    st["bad_reroll"] = []
    for t in range(tries):
        bp = f"A close-up product shot of a {prod} in the foreground, {p}"
        img = LLM.image(IMG_MODEL, bp, task="survey_gen", seed=SEED + 1 + t, width=1024, height=1024)
        name = f"bad_r{t + 1}"
        (sdir(i) / f"{name}.png").write_bytes(img)
        c = {"prompt": bp, "seed": SEED + 1 + t}
        c["verify"] = verify_one(st, name, sdir(i) / f"{name}.png", bp)
        c["judge"] = judge_one(p, sdir(i) / f"{name}.png")
        st["bad_reroll"].append(c); save(i, st)
        if survey_bad(c): break


def do_reroll_base(i, tries: int = 2):
    st = load(i)
    if not (st["cond"]["base"].get("judge") or {}).get("brands") or st.get("base_reroll"): return
    st["base_reroll"] = []
    for t in range(tries):
        img = LLM.image(IMG_MODEL, st["prompt"], task="survey_gen", seed=SEED + 10 + t, width=1024, height=1024)
        name = f"base_r{t + 1}"
        (sdir(i) / f"{name}.png").write_bytes(img)
        c = {"prompt": st["prompt"], "seed": SEED + 10 + t, "judge": judge_one(st["prompt"], sdir(i) / f"{name}.png")}
        st["base_reroll"].append(c); save(i, st)
        if not c["judge"].get("brands"): break


# ------------------------------------------------------------------ loop (feedback repair)
def do_loop(i, which: str, max_extra: int = 2):
    st = load(i)
    if st.get("skip"): return
    start = st["cond"].get(which)
    if not start or not start.get("verify") or st.get("loop"):
        return
    if any(survey_good(st, c) for k, c in st["cond"].items() if k not in ("base", "bad")):
        return
    model = SONNET if which == "v4s" else FLASH
    base = (sdir(i) / "base.png").read_bytes()
    prev, prev_img, prev_v, prev_j = start["prompt"], (sdir(i) / f"{which}.png").read_bytes(), start["verify"], start.get("judge")
    st["loop"] = []
    for j in range(1, max_extra + 1):
        payload = {"action": "regenerate", "original_prompt": st["prompt"], "prompt_sent": prev,
                   "brand": CAT[st["brand"]]["display"], "plan": st["plan"].get("chosen"),
                   "issues": prev_v["issues"], "fix_targets": [x["code"] for x in prev_v["issues"]],
                   "metrics": prev_v["metrics"], "inspector_fix_suggestion": prev_v["informed"].get("fix_suggestion"),
                   "target_model": "instruction-following closed image model (Gemini 3.1 Flash Image): it follows "
                                   "relations, concealment cues ('partly behind the ...') and inventory lists; the "
                                   "session-2 FLUX notes about distance words do not apply",
                   "acceptance": "not blatant and not right in front of the viewer; brand still identifiable",
                   "fresh_viewer_judge": {k: (prev_j or {}).get(k) for k in ("ad_likelihood", "brands", "noticed_first", "brand_naturalness")},
                   "why_rejected": ("brand not identifiable to a fresh viewer" if not brand_seen(st, prev_j) else
                                    "a fresh viewer rates it as an obvious ad (ad_likelihood >= 4)" if (prev_j or {}).get("ad_likelihood", 0) >= 4 else
                                    "verifier issues listed above")}
        fb = LLM.chat_json(model, load_prompt("repair"), payload, task="survey_repair", temperature=0.2,
                           images=[prev_img]).get("revision_feedback", "")
        out = rewrite(LLM, model, st["prompt"], st["brand"], CAT, st["plan"], previous=prev, feedback=fb,
                      prompt_name="rewriter_v4", max_words=18, max_other=24, baseline_image=base)
        newp = out.get("rewritten_prompt") or prev
        img = gen_image(newp)
        (sdir(i) / f"loop{j}.png").write_bytes(img)
        a = {"prompt": newp, "feedback": fb, "rewrite": out, "from": which}
        st["loop"].append(a)
        save(i, st)
        a["verify"] = verify_one(st, f"loop{j}", sdir(i) / f"loop{j}.png", newp)
        a["judge"] = judge_one(st["prompt"], sdir(i) / f"loop{j}.png")
        save(i, st)
        if survey_good(st, a):
            break
        prev, prev_img, prev_v, prev_j = newp, img, a["verify"], a["judge"]


def run_all(fn, idxs, workers=6, **kw):
    def safe(i):
        try:
            fn(i, **kw)
        except Exception as e:  # noqa: BLE001
            print(f"[{i:03d}] {fn.__name__} ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(safe, idxs))
    print(f"{fn.__name__} done · session ledger ${LLM.ledger.spent:.2f} total", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["gen", "verify", "judge", "clip", "loop", "reroll", "all"])
    ap.add_argument("--only")
    ap.add_argument("--which", default="v4s")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    idxs = [int(x) for x in a.only.split(",") if x.strip()] if a.only else list(range(len(PROMPTS)))
    OUT.mkdir(parents=True, exist_ok=True)
    if a.phase in ("gen", "all"): run_all(do_gen, idxs, a.workers)
    if a.phase in ("verify", "all"): run_all(do_verify, idxs, a.workers)
    if a.phase in ("judge", "all"): run_all(do_judge, idxs, a.workers)
    if a.phase in ("clip", "all"): run_all(do_clip, idxs, 1)
    if a.phase == "loop": run_all(do_loop, idxs, a.workers, which=a.which)
    if a.phase == "reroll":
        run_all(do_reroll_bad, idxs, a.workers)
        run_all(do_reroll_base, idxs, a.workers)
