"""Session 3b: simple vs surface-specific rewriter, for the survey set (Jovan, 2026-10-04).

Per prompt (Gemini 3.1 Flash-Lite Image, 1:1, seed 7):
  base     original prompt (reused from runs/survey-closed: brand-free, non-letterboxed seed)
  simple   "<original>, with <product>"                  e.g. "..., with a can of Coke"
  surface  "<original>, with <product> <surface>"        e.g. "..., with a can of Coke on the cafe table"
One LLM call (prompts/placement_phrase.md, sees the base image) writes <product> and <surface>, so the two
rewrites differ only by the surface clause. If `surface` fails Jovan's calibrated rule, a feedback step picks
a different surface from the failed image and the verifier's issues (<= 2 more tries).
Scoring: Gemini 3.7 Flash verifier (calibrated band), GPT-5.6 Sol blind viewer, CLIPScore vs the original
prompt, and a blind side-by-side GPT judge (base vs each rewrite, order randomised).

    .venv/bin/python scripts/survey_surface.py prep|phrase|gen|score|loop|pairwise|summary|export
"""
from __future__ import annotations

import csv
import json
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import survey_bench as sb  # noqa: E402
from adplace.__main__ import make_llm  # noqa: E402
from adplace.pipeline import load_prompt  # noqa: E402

sb.LLM = make_llm(sb.CFG, False, run_id="survey-surface")      # own run id: fresh per-run image cap
LLM, CAT, FLASH, JUDGE = sb.LLM, sb.CAT, sb.FLASH, sb.JUDGE
SRC = sb.OUT
OUT = ROOT / "runs" / "survey-surface"
ALIASES = {"coca-cola": ["coca-cola", "coke"], "apple": ["apple", "macbook", "iphone", "airpods", "ipad"],
           "xbox": ["xbox"], "red-bull": ["red bull", "redbull"], "nike": ["nike", "swoosh"], "pepsi": ["pepsi"],
           "starbucks": ["starbucks"], "ikea": ["ikea"], "adidas": ["adidas"], "samsung": ["samsung"],
           "mcdonalds": ["mcdonald"]}


def norm(x): return str(x).lower().replace("-", "").replace(" ", "").replace("'", "")


def brand_named(brand_key, names) -> bool:
    keys = [norm(a) for a in ALIASES.get(brand_key, [CAT[brand_key]["display"]])]
    return any(k in norm(n) for n in names or [] for k in keys)


def d(i): return OUT / f"{i:03d}"


def load(i):
    p = d(i) / "state.json"
    return json.loads(p.read_text()) if p.exists() else None


def save(i, st):
    d(i).mkdir(parents=True, exist_ok=True)
    (d(i) / "state.json").write_text(json.dumps(st, indent=1))


def ok(c) -> bool:
    """Usable placement: passes Jovan's calibrated rule (implies the verifier found a recognisable product)."""
    return bool((c.get("verify") or {}).get("accepted"))


def idxs():
    return sorted(int(p.name) for p in OUT.glob("[0-9][0-9][0-9]") if load(int(p.name)))


# ------------------------------------------------------------------ prep: reuse base images
def prep():
    sel = {s["idx"]: s for s in json.loads((SRC / "selection.json").read_text())}
    for i in range(len(sb.PROMPTS)):
        if load(i):
            continue
        src = SRC / f"{i:03d}" / "state.json"
        if not src.exists():
            continue
        s = json.loads(src.read_text())
        if s.get("skip") or "plan" not in s:
            continue
        cands = [("base", s["cond"].get("base"))] + [(f"base_r{j}", c) for j, c in enumerate(s.get("base_reroll", []), 1)]
        cands = [(n, c) for n, c in cands if c and (SRC / f"{i:03d}" / f"{n}.png").exists()]
        if i in sel:
            name = sel[i]["base"]
        elif cands:
            name = min(cands, key=lambda nc: (bool(nc[1].get("letterbox")), len((nc[1].get("judge") or {}).get("brands", []))))[0]
        else:
            name = None
        d(i).mkdir(parents=True, exist_ok=True)
        st = {"idx": i, "prompt": s["prompt"], "brand": s["brand"], "plan": s["plan"], "cond": {}}
        if name:
            shutil.copy(SRC / f"{i:03d}" / f"{name}.png", d(i) / "base.png")
            bc = dict(cands)[name]
            st["cond"]["base"] = {"prompt": s["prompt"], "source": f"survey-closed/{name}", "judge": bc.get("judge"),
                                  "clip": bc.get("clip")}
        else:
            (d(i) / "base.png").write_bytes(sb.gen_image(s["prompt"]))
            st["cond"]["base"] = {"prompt": s["prompt"], "source": "generated"}
        save(i, st)
    print("prepared", idxs())


# ------------------------------------------------------------------ phrase
def phrase(i):
    st = load(i)
    if st.get("phrase"):
        return
    b = CAT[st["brand"]]
    payload = {"original_prompt": st["prompt"], "brand": b["display"], "catalogue_forms": b["forms"],
               "fits": b.get("fits"), "avoid": b.get("avoid")}
    out = LLM.chat_json(FLASH, load_prompt("placement_phrase"), payload, task="placement_phrase", temperature=0.2,
                        images=[(d(i) / "base.png").read_bytes()])
    st["phrase"] = out
    save(i, st)


# ------------------------------------------------------------------ gen
def gen(i):
    st = load(i)
    ph = st["phrase"]
    prompts = {"simple": f'{st["prompt"]}, with {ph["product"]}',
               "surface": f'{st["prompt"]}, with {ph["product"]} {ph["surface"]}'}
    for k, p in prompts.items():
        if k in st["cond"] and (d(i) / f"{k}.png").exists():
            continue
        (d(i) / f"{k}.png").write_bytes(sb.gen_image(p))
        st["cond"][k] = {"prompt": p, "surface": ph["surface"] if k == "surface" else None}
        save(i, st)


# ------------------------------------------------------------------ score
def score_one(st, key, c):
    img = d(st["idx"]) / f"{key}.png"
    if key != "base" and not c.get("verify"):
        c["verify"] = sb.verify_one(st, key, img, c["prompt"])
    if not c.get("judge"):
        c["judge"] = sb.judge_one(st["prompt"], img)
    if c.get("clip") is None:
        from adplace.clipscore import clip_cos
        with CLIP_LOCK:                      # one MPS model, not thread-safe
            c["clip"] = clip_cos(img, st["prompt"])


import threading  # noqa: E402
CLIP_LOCK = threading.Lock()


def items(st):
    yield from st["cond"].items()
    for j, a in enumerate(st.get("loop", []), 1):
        yield f"loop{j}", a


def score(i):
    st = load(i)
    for k, c in items(st):
        score_one(st, k, c)
        save(i, st)


# ------------------------------------------------------------------ loop: feedback picks a new surface
FIX = """The surface-specific rewrite below produced an image that failed inspection. You see that image.
Pick ONE different placement surface for the same product, following the same rules as before (a specific place or
object that belongs in this scene, not the focal subject's hands/face, positive wording, no distance/size/blur/hidden
words). Use the issues: TOO_PROMINENT means pick a surface further from the camera or less central (a side table,
a shelf, a bench further along, a bag on the floor); ABSENT means pick a surface where the item would be clearly
visible; ON_SUBJECT means move it off the main subject; SCENE_LOSS/FIDELITY_LOSS means pick something less disruptive.
Reply with JSON only: {"surface": "", "why": ""}"""


def loop(i, max_extra=2):
    st = load(i)
    c = st["cond"].get("surface")
    if not c or ok(c) or st.get("loop"):
        return
    st["loop"] = []
    prev, tried = c, [c["surface"]]
    for j in range(1, max_extra + 1):
        payload = {"original_prompt": st["prompt"], "product": st["phrase"]["product"], "failed_surface": prev["surface"],
                   "issues": prev["verify"]["issues"], "surfaces_already_tried": tried,
                   "suggested_alternatives": st["phrase"].get("alternative_surfaces"),
                   "inspector_fix_suggestion": prev["verify"]["informed"].get("fix_suggestion")}
        fx = LLM.chat_json(FLASH, FIX, payload, task="surface_fix", temperature=0.2,
                           images=[(d(i) / ("surface.png" if j == 1 else f"loop{j - 1}.png")).read_bytes()])
        surf = fx.get("surface") or (st["phrase"].get("alternative_surfaces") or ["on a nearby table"])[min(j - 1, 1)]
        p = f'{st["prompt"]}, with {st["phrase"]["product"]} {surf}'
        (d(i) / f"loop{j}.png").write_bytes(sb.gen_image(p))
        a = {"prompt": p, "surface": surf, "why": fx.get("why")}
        st["loop"].append(a)
        save(i, st)
        score_one(st, f"loop{j}", a)
        save(i, st)
        if ok(a):
            break
        prev = a
        tried.append(surf)


def final_surface(st):
    """The surface-specific image the loop ended on (first accepted, else the last try)."""
    chain = [("surface", st["cond"]["surface"])] + [(f"loop{j}", a) for j, a in enumerate(st.get("loop", []), 1)]
    for k, c in chain:
        if ok(c):
            return k, c
    return chain[-1]


# ------------------------------------------------------------------ surface-short: relational reference
SHORT = """Rewrite the placement surface below as the SHORTEST natural reference to a place the people in the scene are
already using or sitting at, 2-5 words, possessive or relational where it fits: "on their table", "beside her on the
bench", "in the cup holder", "among their snacks", "on the counter". Do not describe the surface (no materials, shapes,
adjectives like "round", "wooden", "marble"), do not introduce a new object, no distance/size/blur/hidden words.
Reply with JSON only: {"surface_short": ""}"""


def short(i):
    st = load(i)
    if "surface_short" in st["cond"] and (d(i) / "surface_short.png").exists():
        return
    if not st["phrase"].get("surface_short"):
        out = LLM.chat_json(FLASH, SHORT, {"original_prompt": st["prompt"], "product": st["phrase"]["product"],
                                           "surface": st["phrase"]["surface"]}, task="surface_short", temperature=0.2)
        st["phrase"]["surface_short"] = out.get("surface_short") or st["phrase"]["surface"]
        save(i, st)
    p = f'{st["prompt"]}, with {st["phrase"]["product"]} {st["phrase"]["surface_short"]}'
    (d(i) / "surface_short.png").write_bytes(sb.gen_image(p))
    st["cond"]["surface_short"] = {"prompt": p, "surface": st["phrase"]["surface_short"]}
    save(i, st)
    score_one(st, "surface_short", st["cond"]["surface_short"])
    save(i, st)


def letterboxed(path) -> bool:
    from PIL import Image, ImageStat
    im = Image.open(path).convert("L")
    w, h = im.size
    for box in ((0, 0, w, int(h * .04)), (0, int(h * .96), w, h)):
        s = ImageStat.Stat(im.crop(box))
        if s.stddev[0] < 4 and (s.mean[0] > 235 or s.mean[0] < 15):
            return True
    return False


# ------------------------------------------------------------------ pairwise (blind, side by side)
def pairwise(i):
    from survey_select import PAIR_PROMPT
    st = load(i)
    rnd = random.Random(1000 + i)
    for kind, (key, c) in (("simple", ("simple", st["cond"]["simple"])), ("surface", final_surface(st))):
        if st.get("pair_" + kind):
            continue
        flip = rnd.random() < 0.5
        a, b = (d(i) / "base.png").read_bytes(), (d(i) / f"{key}.png").read_bytes()
        r = LLM.chat_json(JUDGE, PAIR_PROMPT, f"Prompt: {st['prompt']!r}", task="survey_pairwise", temperature=0.0,
                          images=[b, a] if flip else [a, b])
        inj = "1" if flip else "2"
        r.update(image=key, injected_position=inj, detected=r.get("which_has_ad") == inj,
                 prefers_injected=r.get("prefer") == inj)
        st["pair_" + kind] = r
        save(i, st)


def run(fn, ix, workers=6):
    def safe(i):
        try:
            fn(i)
        except Exception as e:  # noqa: BLE001
            print(f"[{i:03d}] {fn.__name__} ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(safe, ix))
    print(f"{fn.__name__} done · ledger ${LLM.ledger.spent:.2f}", flush=True)


# ------------------------------------------------------------------ summary / export
def summary():
    import statistics as stt
    rows = {"base": [], "simple": [], "surface": [], "surface_short": [], "surface+loop": []}
    out = []
    for i in idxs():
        st = load(i)
        if "simple" not in st["cond"] or not st["cond"]["simple"].get("verify"):
            continue
        rows["base"].append((st, st["cond"]["base"]))
        rows["simple"].append((st, st["cond"]["simple"]))
        rows["surface"].append((st, st["cond"]["surface"]))
        if st["cond"].get("surface_short", {}).get("verify"):
            rows["surface_short"].append((st, st["cond"]["surface_short"]))
        rows["surface+loop"].append((st, final_surface(st)[1]))
    m = lambda xs: round(stt.mean(xs), 3) if xs else None
    res = {}
    for k, R in rows.items():
        J = [c.get("judge") or {} for _, c in R]
        V = [c.get("verify") for _, c in R if c.get("verify")]
        res[k] = {"n": len(R), "accepted": sum(ok(c) for _, c in R) if k != "base" else None,
                  "brand_named": sum(brand_named(s["brand"], (c.get("judge") or {}).get("brands")) for s, c in R),
                  "judge_ad": m([j.get("ad_likelihood") for j in J if j.get("ad_likelihood")]),
                  "adherence": m([j.get("adherence") for j in J if j.get("adherence")]),
                  "quality": m([j.get("quality") for j in J if j.get("quality")]),
                  "naturalness": m([j.get("brand_naturalness") for j in J if j.get("brand_naturalness")]),
                  "legibility": m([v["informed"].get("legibility") or 0 for v in V]) if V else None,
                  "blind_ad": m([(v.get("blind") or {}).get("ad_suspicion") or 0 for v in V]) if V else None,
                  "too_prominent": sum(any(x["code"] == "TOO_PROMINENT" for x in v["issues"]) for v in V) if V else None,
                  "clip": m([c["clip"] for _, c in R if c.get("clip") is not None])}
    for k in ("simple", "surface"):
        P = [load(i).get("pair_" + k) for i in idxs() if load(i).get("pair_" + k)]
        res[k]["pair_detected"] = f"{sum(p['detected'] for p in P)}/{len(P)}" if P else None
        res[k]["pair_prefers"] = f"{sum(p['prefers_injected'] for p in P)}/{len(P)}" if P else None
    res["surface+loop"]["pair_detected"] = res["surface"]["pair_detected"]
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, v)


def export():
    dest = ROOT / "survey"
    if dest.exists() and not (ROOT / "survey_v1_good-bad").exists():
        shutil.move(str(dest), ROOT / "survey_v1_good-bad")      # keep the earlier (superseded) set
    elif dest.exists():
        shutil.rmtree(dest)
    (dest / "images").mkdir(parents=True)
    rnd = random.Random(2026)
    rows, n = [], 0
    for i in idxs():
        st = load(i)
        if "simple" not in st["cond"]:
            continue
        sk, sc = final_surface(st)
        if not ok(sc):
            continue          # the survey needs a surface-specific placement that passes the rule
        n += 1
        letters = ["A", "B", "C"]
        rnd.shuffle(letters)
        for role, key, c, letter in (("no-inject", "base", st["cond"]["base"], letters[0]),
                                     ("simple", "simple", st["cond"]["simple"], letters[1]),
                                     ("surface", sk, sc, letters[2])):
            name = f"S{n:02d}_{letter}.jpg"
            shutil.copy(d(i) / f"{key}.png", dest / "images" / name)
            j, v = c.get("judge") or {}, c.get("verify") or {}
            rows.append({"item": f"S{n:02d}", "file": name, "condition": role, "user_prompt": st["prompt"],
                         "brand": "" if role == "no-inject" else CAT[st["brand"]]["display"],
                         "prompt_sent_to_model": c["prompt"], "source": f"runs/survey-surface/{i:03d}/{key}.png",
                         "passes_rule": ok(c) if role != "no-inject" else "", "judge_ad_likelihood": j.get("ad_likelihood"),
                         "judge_brands": "; ".join(j.get("brands") or []),
                         "legibility": (v.get("informed") or {}).get("legibility"), "clipscore": c.get("clip"),
                         "model": sb.IMG_MODEL})
    with (dest / "key.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (dest / "README.txt").write_text(
        "Survey image set v2 (adplace, 2026-10-04): per item, the user's prompt rendered three ways —\n"
        "no-inject (original prompt), simple ('..., with a can of Coke') and surface ('..., with a can of Coke on the cafe table').\n"
        "images/S<item>_<A|B|C>.jpg: letters randomly assigned per item. key.csv: condition, prompts, scores (keep from participants).\n"
        "Model: Gemini 3.1 Flash-Lite Image (OpenRouter), 1:1. The three images are separate renders of the same prompt.\n")
    print(f"exported {n} items -> {dest}")


if __name__ == "__main__":
    ph = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    if ph == "prep":
        prep()
    elif ph in ("phrase", "gen", "score", "loop", "pairwise", "short"):
        run({"phrase": phrase, "gen": gen, "score": score, "loop": loop, "pairwise": pairwise, "short": short}[ph], idxs())
    elif ph == "letterbox":
        for i in idxs():
            bad = [f.stem for f in sorted(d(i).glob("*.png")) if letterboxed(f)]
            if bad:
                print(i, bad)
    elif ph == "summary":
        summary()
    elif ph == "export":
        export()
