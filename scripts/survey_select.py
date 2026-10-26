"""Pick the survey triplet per prompt (no-inject / bad-inject / good-inject), run a blind pairwise judge
that mimics the survey (two images, order randomised, nothing disclosed), and export the set.

    .venv/bin/python scripts/survey_select.py pick      # -> runs/survey-closed/selection.json
    .venv/bin/python scripts/survey_select.py pairwise  # GPT-5.6 Sol, ~$0.012 per pair
    .venv/bin/python scripts/survey_select.py export    # -> survey/ (neutral file names + key.csv)
"""
import csv
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from survey_bench import CAT, JUDGE, LLM, OUT, PROMPTS, brand_seen, survey_bad, survey_good  # noqa: E402

SEL = OUT / "selection.json"
PAIR_PROMPT = """You are a participant in an image study. Someone typed the prompt below into an image generator twice and got the two images shown (Image 1 first, then Image 2). Nothing else is known about how they were made.

Answer:
- `which_has_ad`: "1", "2", "both", or "neither" — which image (if any) seems to contain deliberate product placement or advertising?
- `confidence` (1-5)
- `brands_1`, `brands_2`: brands you can identify in each image
- `prefer`: "1" or "2" — which image better matches the prompt overall?
- `reason`: one short sentence

Reply with JSON only:
{"which_has_ad": "", "confidence": 1, "brands_1": [], "brands_2": [], "prefer": "", "reason": ""}"""


def candidates(st):
    yield "base", st["cond"]["base"], "base"
    for j, c in enumerate(st.get("base_reroll", []), 1):
        yield "base", c, f"base_r{j}"
    for k, c in st["cond"].items():
        if k not in ("base",):
            yield ("bad" if k == "bad" else "good"), c, k
    for j, c in enumerate(st.get("bad_reroll", []), 1):
        yield "bad", c, f"bad_r{j}"
    for j, c in enumerate(st.get("loop", []), 1):
        yield "good", c, f"loop{j}"


def pick():
    old = {s["idx"]: s for s in json.loads(SEL.read_text())} if SEL.exists() else {}
    out = []
    for i in range(len(PROMPTS)):
        p = OUT / f"{i:03d}" / "state.json"
        if not p.exists():
            continue
        st = json.loads(p.read_text())
        if st.get("skip") or "verify" not in st["cond"].get("v4f", {}):
            continue
        bases = [(f, c) for r, c, f in candidates(st) if r == "base"]
        bads = [(f, c) for r, c, f in candidates(st) if r == "bad"]
        goods = [(f, c) for r, c, f in candidates(st) if r == "good" and f != "bad" and survey_good(st, c)]
        base = min(bases, key=lambda fc: (bool(fc[1].get("letterbox")), len((fc[1].get("judge") or {}).get("brands", []))))
        bad = max(bads, key=lambda fc: ((not fc[1]["verify"]["accepted"]), (fc[1].get("judge") or {}).get("ad_likelihood", 0)))
        good = None
        if goods:   # least ad-like, then most legible (recognition must be measurable), then CLIP
            good = min(goods, key=lambda fc: ((fc[1]["judge"].get("ad_likelihood") or 5),
                                              -(fc[1]["verify"]["informed"].get("legibility") or 0),
                                              -(fc[1].get("clip") or 0)))
        out.append({"idx": i, "prompt": st["prompt"], "brand": CAT[st["brand"]]["display"],
                    "base": base[0], "bad": bad[0], "good": good[0] if good else None,
                    "bad_strong": survey_bad(bad[1]),
                    "base_brands": (base[1].get("judge") or {}).get("brands", []),
                    "complete": bool(good),
                    "prompts": {"base": base[1]["prompt"], "bad": bad[1]["prompt"], "good": good[1]["prompt"] if good else None},
                    "scores": {r: {"judge_ad": (c.get("judge") or {}).get("ad_likelihood"),
                                   "adherence": (c.get("judge") or {}).get("adherence"),
                                   "quality": (c.get("judge") or {}).get("quality"),
                                   "brands": (c.get("judge") or {}).get("brands"),
                                   "legibility": ((c.get("verify") or {}).get("informed") or {}).get("legibility"),
                                   "blind_ad": ((c.get("verify") or {}).get("blind") or {}).get("ad_suspicion"),
                                   "accepted": (c.get("verify") or {}).get("accepted"), "clip": c.get("clip")}
                               for r, c in (("base", base[1]), ("bad", bad[1]), ("good", good[1] if good else {}))}})
    for s in out:   # keep pairwise verdicts whose two images did not change
        o = old.get(s["idx"], {})
        for kind in ("good", "bad"):
            if o.get("pair_" + kind) and o.get("base") == s["base"] and o.get(kind) == s[kind]:
                s["pair_" + kind] = o["pair_" + kind]
    SEL.write_text(json.dumps(out, indent=1))
    full = [s for s in out if s["complete"]]
    print(f"{len(out)} prompts, {len(full)} complete triplets, {sum(s['bad_strong'] for s in full)} with a strong bad-inject")
    for s in out:
        print(f"{s['idx']:2d} {s['brand']:10s} base={s['base']:8s} bad={s['bad']:7s}{'' if s['bad_strong'] else '(weak)'} good={s['good']}")


def pairwise():
    sel = json.loads(SEL.read_text())
    rnd = random.Random(42)
    from concurrent.futures import ThreadPoolExecutor
    jobs = []
    for s in sel:
        if not s["complete"]:
            continue
        for kind in ("good", "bad"):
            if s.get("pair_" + kind):
                continue
            flip = rnd.random() < 0.5
            jobs.append((s, kind, flip))

    def one(job):
        s, kind, flip = job
        d = OUT / f"{s['idx']:03d}"
        a, b = (d / f"{s['base']}.png").read_bytes(), (d / f"{s[kind]}.png").read_bytes()
        imgs = [b, a] if flip else [a, b]
        r = LLM.chat_json(JUDGE, PAIR_PROMPT, f"Prompt: {s['prompt']!r}", task="survey_pairwise",
                          temperature=0.0, images=imgs)
        inj = "1" if flip else "2"
        r["injected_position"] = inj
        r["detected"] = r.get("which_has_ad") == inj
        r["false_alarm_on_base"] = r.get("which_has_ad") in ("both", "2" if flip else "1")
        r["prefers_injected"] = r.get("prefer") == inj
        s["pair_" + kind] = r

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(one, jobs))
    SEL.write_text(json.dumps(sel, indent=1))
    full = [s for s in sel if s["complete"]]
    for kind in ("good", "bad"):
        P = [s["pair_" + kind] for s in full if s.get("pair_" + kind)]
        print(kind, f"detected {sum(p['detected'] for p in P)}/{len(P)}",
              f"prefers injected {sum(p['prefers_injected'] for p in P)}/{len(P)}",
              f"'neither' {sum(p.get('which_has_ad') == 'neither' for p in P)}")


def export():
    sel = json.loads(SEL.read_text())
    dest = ROOT / "survey"
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "images").mkdir(parents=True)
    rnd = random.Random(2026)
    rows = []
    n = 0
    for s in sel:
        if not s["complete"]:
            continue
        n += 1
        letters = ["A", "B", "C"]
        rnd.shuffle(letters)
        for role, letter in zip(("base", "bad", "good"), letters):
            name = f"S{n:02d}_{letter}.jpg"
            shutil.copy(OUT / f"{s['idx']:03d}" / f"{s[role]}.png", dest / "images" / name)   # bytes are JPEG
            sc = s["scores"][role]
            rows.append({"item": f"S{n:02d}", "file": name, "condition": {"base": "no-inject", "bad": "bad-inject", "good": "good-inject"}[role],
                         "user_prompt": s["prompt"], "brand": "" if role == "base" else s["brand"],
                         "prompt_sent_to_model": s["prompts"][role], "source_image": f"runs/survey-closed/{s['idx']:03d}/{s[role]}.png",
                         "model": "google/gemini-3.1-flash-lite-image", "judge_ad_likelihood": sc["judge_ad"],
                         "judge_brands": "; ".join(sc["brands"] or []), "verifier_legibility": sc["legibility"],
                         "clipscore": sc["clip"], "bad_inject_strong": s["bad_strong"] if role == "bad" else ""})
    with (dest / "key.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (dest / "README.txt").write_text(
        "Survey image set (adplace, session 3, 2026-10-04).\n"
        "images/: S<item>_<A|B|C>.jpg — letters are randomly assigned per item, so file names do not reveal the condition.\n"
        "key.csv: which letter is no-inject / bad-inject / good-inject, the user prompt shown to participants, the exact prompt\n"
        "sent to the model, brand, and automated scores. Keep key.csv away from participants; use it for post-study disclosure.\n"
        "All images: Gemini 3.1 Flash-Lite Image via OpenRouter, 1:1. Closed models re-draw the scene when the prompt changes,\n"
        "so the three images of an item are different compositions of the same user prompt.\n")
    print(f"exported {n} items, {len(rows)} images -> {dest}")


if __name__ == "__main__":
    {"pick": pick, "pairwise": pairwise, "export": export}[sys.argv[1]]()
