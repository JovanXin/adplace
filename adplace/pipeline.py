"""Stage 0-2: keyword gate, context classifier, placement planner + rewriter."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

PROMPTS = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPTS / f"{name}.md").read_text()


def load_catalog(path: Path | None = None) -> dict:
    return yaml.safe_load((path or Path(__file__).parent / "brands.yaml").read_text())


# --------------------------------------------------------------------------
# Stage 0: cheap keyword gate (Thesis A 5.2) - avoids an LLM call on obvious cases.
# Only ever returns EXCLUDE or None (= "ask the classifier"); it never approves.
# --------------------------------------------------------------------------
_GATE = {
    "minors": r"\b(child|children|kid|kids|toddler|baby|babies|infant|schoolgirl|schoolboy|kindergarten|nursery|playground|preschool)\b",
    "tragedy": r"\b(funeral|grave|graveyard|cemetery|corpse|dead body|mourning|grief|disaster|tsunami|earthquake victims?|genocide|holocaust|9/11)\b",
    "violence": r"\b(gore|bloody|massacre|shooting|stabbing|murder|war crime|torture|terrorist)\b",
    "medical": r"\b(hospital bed|icu|surgery|chemotherapy|cancer|patient|overdose|suicide|self[- ]harm)\b",
    "religion": r"\b(church|mosque|synagogue|temple|cathedral|chapel|prayer|praying|jesus|allah|buddha|crucifix)\b",
    "politics": r"\b(election|ballot|protest|campaign rally|president|prime minister|parliament|senator)\b",
    "nsfw": r"\b(nude|naked|nsfw|explicit|topless|lingerie|porn)\b",
}
# Brands commonly named in prompts (existing-brand exclusion); extend as needed.
_KNOWN_BRANDS = r"\b(coca[- ]?cola|coke|pepsi|nike|adidas|apple|iphone|macbook|samsung|starbucks|mcdonald'?s|red bull|xbox|playstation|ikea|heineken|gucci|louis vuitton|tesla|bmw|mercedes|ferrari|lego|disney|marvel|pokemon|nintendo|google|amazon|supreme)\b"
_APPLE_FRUIT = r"\bapples?\b(?!\s*(computer|laptop|logo|watch|store|macbook|iphone))"


def keyword_gate(prompt: str) -> dict | None:
    p = prompt.lower()
    flags = [k for k, rx in _GATE.items() if re.search(rx, p)]
    brand_hits = re.findall(_KNOWN_BRANDS, p)
    # "apple" the fruit is not a brand cue (Thesis B edge case)
    brand_hits = [b for b in brand_hits if not (b == "apple" and re.search(_APPLE_FRUIT, p))]
    if brand_hits:
        flags.append("existing_brand")
    if flags:
        return {"decision": "EXCLUDE", "exclude_flags": flags, "brands": [],
                "reason": f"keyword gate: {', '.join(flags)}", "source": "keyword_gate"}
    return None


# --------------------------------------------------------------------------
# Stage 1: context classifier
# --------------------------------------------------------------------------
def classify(llm, model: str, prompt: str, catalog: dict, use_gate: bool = True) -> dict:
    if use_gate:
        g = keyword_gate(prompt)
        if g:
            return g
    compact = {k: {f: v[f] for f in ("display", "category", "fits", "avoid", "portrayal_rules",
                                        "restricted", "render_reliability")} for k, v in catalog.items()}
    out = llm.chat_json(model, load_prompt("classifier"),
                        {"user_prompt": prompt, "catalogue": compact}, task="classify")
    out["source"] = "llm"
    out["decision"] = str(out.get("decision", "SKIP")).upper()
    # Hard guardrails the LLM is not allowed to override.
    valid = [b for b in out.get("brands", []) if b.get("brand_key") in catalog]
    out["brands"] = sorted(valid, key=lambda b: -float(b.get("fit", 0)))
    if out["decision"] == "INJECT" and not out["brands"]:
        out["decision"] = "SKIP"
        out["reason"] += " [no valid catalogue brand returned]"
    return out


# --------------------------------------------------------------------------
# Stage 2a: placement planner (scene analysis -> 3 candidates -> choose)
# --------------------------------------------------------------------------
WEIGHTS = {"subtlety": 0.30, "preservation": 0.30, "congruence": 0.20,
           "recognisability": 0.10, "renderability": 0.10}


def _overlaps_focal(anchor: str, focal: str) -> bool:
    """True when the anchor names the focal subject (the Thesis B 'Nike on the jacket' failure)."""
    a, f = set(_content_words(anchor or "")), set(_content_words(focal or ""))
    generic = {"table", "scene", "background", "person", "people", "man", "woman", "side", "edge"}
    return bool((a & f) - generic)


def score_candidate(c: dict, focal: str = "") -> float:
    s = c.get("scores", {})
    total = sum(WEIGHTS[k] * float(s.get(k, 0)) for k in WEIGHTS)
    if float(s.get("recognisability", 0)) < 3:     # invisible placements are useless
        total -= 1.0
    if "centre" in str(c.get("frame_region", "")).lower() or "center" in str(c.get("frame_region", "")).lower():
        total -= 0.5
    if c.get("depth_plane") == "foreground":
        total -= 0.3
    if c.get("strategy") != "incidental_use" and _overlaps_focal(c.get("anchor", ""), focal):
        total -= 2.0                                # never brand the thing the user asked for
    return round(total, 3)


def plan(llm, model: str, prompt: str, brand_key: str, catalog: dict, visibility: str,
         avoid_ids: list[int] | None = None, classification: dict | None = None,
         previous: dict | None = None, previous_failures: list[dict] | None = None) -> dict:
    """Plan a placement. With `previous` + `avoid_ids`, re-use the earlier candidates and just
    pick the next best (no LLM call). With `previous_failures`, ask the LLM for fresh ones."""
    if previous and avoid_ids and not previous_failures:
        out = {k: v for k, v in previous.items() if k not in ("chosen", "chosen_id")}
        cands = out.get("candidates", [])
    else:
        b = catalog[brand_key]
        payload = {"user_prompt": prompt, "brand_key": brand_key, "brand": b,
                   "visibility_target": visibility}
        if classification:
            payload["prompt_specificity"] = classification.get("prompt_specificity")
            payload["candidate_surfaces"] = classification.get("candidate_surfaces")
        if previous_failures:
            payload["previous_failures"] = previous_failures
        out = llm.chat_json(model, load_prompt("planner"), payload, task="plan", temperature=0.4)
        cands = out.get("candidates", [])
    focal = (out.get("scene") or {}).get("focal_subject", "")
    for c in cands:
        c["weighted_score"] = score_candidate(c, focal)
    # We re-rank deterministically rather than trusting the model's own pick,
    # so the choice is reproducible and explainable in the thesis.
    pool = [c for c in cands if c.get("id") not in (avoid_ids or [])] or cands
    if pool:
        best = max(pool, key=lambda c: c["weighted_score"])
        out["llm_chosen_id"] = out.get("chosen_id")
        out["chosen_id"] = best.get("id")
        out["chosen"] = best
    return out


# --------------------------------------------------------------------------
# Stage 2b: rewriter + deterministic lint
# --------------------------------------------------------------------------
BANNED = ["advert", "advertisement", "sponsored", "branding", "prominent", "prominently",
          "product shot", "hero", "iconic", "refreshing", "featuring", "showcase",
          "clearly visible", "in focus", "close-up of the"]
STOP = set("a an the of in on at with and or to for by from is are be as into over under near its it this that "
           "very highly detailed 4k 8k hd".split())


def _content_words(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", s.lower()) if w not in STOP and len(w) > 2]


NEGATION = r"\b(not|no|never|without|avoid|don'?t|doesn'?t|isn'?t)\b"


def inserted_text(original: str, rewritten: str) -> str:
    """Words present in the rewrite but not the original (word-level diff), joined as spans."""
    import difflib
    a, b = original.split(), rewritten.split()
    norm = lambda ws: [re.sub(r"[^\w']", "", w.lower()) for w in ws]  # "beach" == "beach,"
    spans = []
    for op, _, _, j1, j2 in difflib.SequenceMatcher(a=norm(a), b=norm(b), autojunk=False).get_opcodes():
        if op in ("insert", "replace"):
            spans.append(" ".join(b[j1:j2]))
    return " … ".join(spans)


@dataclass
class LintResult:
    ok: bool
    problems: list[str]
    retention: float
    inserted: str = ""
    brand_pos: float = 0.0


def lint(original: str, rewritten: str, brand_display: str, max_words: int = 45,
         max_other: int = 20) -> LintResult:
    probs = []
    orig_words = _content_words(original)
    new = set(_content_words(rewritten))
    kept = sum(1 for w in orig_words if w in new)
    retention = kept / max(1, len(orig_words))
    if retention < 0.9:
        probs.append(f"only {retention:.0%} of the original content words were kept")
    low = rewritten.lower()
    n_brand = len(re.findall(rf"\b{re.escape(brand_display.lower())}\b", low)) - \
        len(re.findall(rf"\b{re.escape(brand_display.lower())}\b", original.lower()))
    if n_brand == 0:
        probs.append("brand not named")
    elif n_brand > 1:
        probs.append(f"brand named {n_brand} times (should be once)")
    for w in BANNED:
        if re.search(rf"\b{re.escape(w)}\b", low):
            probs.append(f"uses banned word '{w}'")
    if len(rewritten) > len(original) + 260 + 8 * max(0, max_other - 20):
        probs.append("insertion too long; image models over-weight long clauses")
    ins = inserted_text(original, rewritten)
    if re.search(r"\b(bottles|cans|cups|bags)\b", ins.lower()):
        probs.append("plural product form introduced (risk of duplicates)")
    neg = re.findall(NEGATION, ins.lower())
    if neg:
        probs.append(f"negation in the inserted text ({', '.join(sorted(set(neg)))}); image models "
                     "ignore it. Rephrase positively (e.g. 'turned away', 'half hidden')")
    m = re.search(rf"\b{re.escape(brand_display.lower())}\b", low)
    brand_pos = m.start() / max(1, len(low)) if m else 0.0
    if m and brand_pos < 0.25 and len(original) > 60:
        probs.append(f"brand appears at {brand_pos:.0%} of the prompt; move the clause after the main "
                     "subject/scene description (early tokens get the most weight)")
    spans = [x.strip() for x in ins.split(" … ") if x.strip()]
    brand_span = next((x for x in spans if brand_display.lower() in x.lower()), ins)
    other = sum(len(x.split()) for x in spans if x is not brand_span)
    if len(brand_span.split()) > max_words:
        probs.append(f"the product phrase is {len(brand_span.split())} words; keep it to {max_words} or fewer "
                     "(every extra word about the product makes the image model feature it more)")
    if other > max_other:
        probs.append(f"{other} words were added elsewhere in the prompt; keep other additions to {max_other} or fewer")
    return LintResult(not probs, probs, round(retention, 3), ins, round(brand_pos, 3))


def rewrite(llm, model: str, prompt: str, brand_key: str, catalog: dict, plan_out: dict,
            previous: str | None = None, feedback: str | None = None, max_fix: int = 2,
            prompt_name: str = "rewriter", max_words: int = 45, max_other: int = 20,
            baseline_image: bytes | None = None, baseline_props: list | None = None) -> dict:
    b = catalog[brand_key]
    payload = {"original_prompt": prompt, "brand": b["display"], "plan": plan_out.get("chosen"),
               "scene": plan_out.get("scene")}
    if previous:
        payload["previous_rewrite"] = previous
    if feedback:
        payload["revision_feedback"] = feedback
    if baseline_props is not None:
        payload["baseline_props"] = baseline_props
    imgs = [baseline_image] if baseline_image is not None else None
    if imgs:
        payload["image"] = "attached: the baseline FLUX drew for the original prompt on the same seed"
    out = llm.chat_json(model, load_prompt(prompt_name), payload, task="rewrite", temperature=0.3, images=imgs)
    history = []
    for _ in range(max_fix):
        lr = lint(prompt, out.get("rewritten_prompt", ""), b["display"], max_words, max_other)
        history.append({"prompt": out.get("rewritten_prompt"), "lint": lr.__dict__})
        if lr.ok:
            break
        payload["previous_rewrite"] = out.get("rewritten_prompt")
        payload["revision_feedback"] = "Fix these problems and change nothing else: " + "; ".join(lr.problems)
        out = llm.chat_json(model, load_prompt(prompt_name), payload, task="rewrite:lint_fix", temperature=0.2,
                            images=imgs)
    lr = lint(prompt, out.get("rewritten_prompt", ""), b["display"], max_words, max_other)
    out["lint"] = lr.__dict__
    out["lint_history"] = history
    out["inserted_computed"] = lr.inserted
    return out
