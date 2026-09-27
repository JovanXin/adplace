"""Stage 4: post-generation verification.

Two VLM passes, then a deterministic judge:
  1. blind   - a naive viewer sees only the ORIGINAL prompt and the image (detectability proxy)
  2. informed - an inspector who knows the brand/plan returns boxes and 1-5 ratings
  3. metrics - area / count / centrality / subject overlap are computed from the boxes here,
               not taken from the VLM, so thresholds are reproducible
  4. judge   - compares metrics against the prominence band in config -> issue codes
"""
from __future__ import annotations

import io
import math
import re

from .pipeline import load_prompt

# Issue codes, most severe first. The loop repairs the first one it can act on.
PRIORITY = ["FIDELITY_LOSS", "SCENE_LOSS", "ARTIFACTS", "ABSENT", "ON_SUBJECT", "DUPLICATES",
            "TOO_PROMINENT", "GARBLED_BRAND", "BAD_INTEGRATION"]


# --------------------------------------------------------------------------
# Box geometry (boxes are [ymin, xmin, ymax, xmax] on a 0-1000 grid)
# --------------------------------------------------------------------------
def _norm_box(b) -> tuple[float, float, float, float] | None:
    try:
        y0, x0, y1, x1 = (float(v) / 1000.0 for v in b)
    except (TypeError, ValueError):
        return None
    y0, y1 = sorted((min(max(y0, 0), 1), min(max(y1, 0), 1)))
    x0, x1 = sorted((min(max(x0, 0), 1), min(max(x1, 0), 1)))
    if y1 - y0 <= 0 or x1 - x0 <= 0:
        return None
    return y0, x0, y1, x1


def _area(b) -> float:
    return (b[2] - b[0]) * (b[3] - b[1])


def _iou(a, b) -> float:
    iy = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ix = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iy * ix
    union = _area(a) + _area(b) - inter
    return inter / union if union else 0.0


def _inside_frac(a, b) -> float:
    """Fraction of box a that lies inside box b."""
    iy = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ix = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return iy * ix / _area(a) if _area(a) else 0.0


def centrality(b) -> float:
    """1 = box centre at frame centre, 0 = at a corner."""
    cy, cx = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return 1 - math.hypot(cy - 0.5, cx - 0.5) / math.hypot(0.5, 0.5)


def box_metrics(product_boxes, subject_box, exclude_boxes=None) -> dict:
    boxes = [nb for nb in (_norm_box(b) for b in (product_boxes or [])) if nb]
    excl = [nb for nb in (_norm_box(b) for b in (exclude_boxes or [])) if nb]
    boxes = [b for b in boxes if all(_iou(b, e) < 0.5 for e in excl)]   # pre-existing brand instances
    # A VLM sometimes returns one box twice; merge near-identical boxes before counting.
    uniq: list = []
    for b in sorted(boxes, key=_area, reverse=True):
        if all(_iou(b, u) < 0.6 for u in uniq):
            uniq.append(b)
    subj = _norm_box(subject_box) if subject_box else None
    if not uniq:
        return {"count": 0, "area": 0.0, "total_area": 0.0, "centrality": 0.0,
                "subject_iou": 0.0, "inside_subject": 0.0,
                "subject_area": round(_area(subj), 4) if subj else None, "largest_box": None}
    big = uniq[0]
    return {
        "count": len(uniq),
        "area": round(_area(big), 5),
        "total_area": round(sum(_area(b) for b in uniq), 5),
        "centrality": round(centrality(big), 3),
        "subject_iou": round(_iou(big, subj), 3) if subj else 0.0,
        "inside_subject": round(_inside_frac(big, subj), 3) if subj else 0.0,
        "subject_area": round(_area(subj), 4) if subj else None,
        "largest_box": [round(v, 4) for v in big],
    }


# --------------------------------------------------------------------------
# Edit drift: how much an edit changed pixels OUTSIDE the product boxes
# --------------------------------------------------------------------------
def edit_drift(before: bytes, after: bytes, boxes_norm: list, pad: float = 0.04,
               thresh: int = 20) -> float:
    from PIL import Image, ImageChops, ImageFilter
    n = 256
    a = Image.open(io.BytesIO(before)).convert("L").resize((n, n)).filter(ImageFilter.GaussianBlur(1))
    b = Image.open(io.BytesIO(after)).convert("L").resize((n, n)).filter(ImageFilter.GaussianBlur(1))
    d = ImageChops.difference(a, b).load()
    masks = [(max(0, int((y0 - pad) * n)), max(0, int((x0 - pad) * n)),
              min(n, int((y1 + pad) * n)), min(n, int((x1 + pad) * n))) for y0, x0, y1, x1 in boxes_norm]
    changed = total = 0
    for y in range(n):
        for x in range(n):
            if any(my0 <= y < my1 and mx0 <= x < mx1 for my0, mx0, my1, mx1 in masks):
                continue
            total += 1
            changed += d[x, y] > thresh
    return round(changed / max(1, total), 4)


# --------------------------------------------------------------------------
# VLM passes
# --------------------------------------------------------------------------
def blind_pass(llm, model: str, image: bytes, original_prompt: str) -> dict:
    return llm.chat_json(model, load_prompt("verify_blind"),
                         f"Prompt the person typed: {original_prompt!r}", task="verify_blind",
                         temperature=0.0, images=[image])


def informed_pass(llm, model: str, image: bytes, ctx: dict, reference: bytes | None = None) -> dict:
    """With `reference` (the same-seed baseline) the inspector also lists baseline props that vanished."""
    if reference is not None:
        ctx = {**ctx, "images": "FIRST = reference (original prompt, same seed, no product); SECOND = image to inspect"}
    return llm.chat_json(model, load_prompt("verify_informed"), ctx, task="verify_informed",
                         temperature=0.0, images=[reference, image] if reference is not None else [image])


# --------------------------------------------------------------------------
# Judge
# --------------------------------------------------------------------------
def _brand_in(brand_display: str, seen: list) -> bool:
    key = re.sub(r"[^a-z]", "", brand_display.lower())
    return any(key and key in re.sub(r"[^a-z]", "", str(s).lower()) for s in seen or [])


def judge(inf: dict, blind: dict | None, m: dict, band: dict, min_fidelity: int,
          min_integration: int, strategy: str = "", drift: float | None = None,
          max_drift: float = 0.05, max_scene_loss: float = 0.5) -> list[dict]:
    issues: list[dict] = []

    def add(code, detail):
        issues.append({"code": code, "detail": detail})

    leg = int(inf.get("legibility") or 0)
    att = int(inf.get("draws_attention") or 1)
    fid = int(inf.get("fidelity_to_original") or 0)
    integ = int(inf.get("integration") or 0)
    susp = int((blind or {}).get("ad_suspicion") or 1)

    # -- presence ---------------------------------------------------------
    if not inf.get("product_found") or m["count"] == 0:
        add("ABSENT", "product not found")
    elif leg < band["min_legibility"]:
        add("ABSENT", f"legibility {leg} < {band['min_legibility']} (unrecognisable)")
    elif m["area"] < band["min_area"]:
        add("ABSENT", f"area {m['area']:.4f} < {band['min_area']} (too small to register)")

    # -- prominence ceiling ----------------------------------------------
    if m["count"]:
        why = []
        if band.get("max_area") is not None and m["area"] > band["max_area"]:
            why.append(f"area {m['area']:.3f} > {band['max_area']}")
        if band.get("max_centrality") is not None and m["centrality"] > band["max_centrality"]:
            why.append(f"centrality {m['centrality']:.2f} > {band['max_centrality']}")
        if leg > band["max_legibility"]:
            why.append(f"legibility {leg} > {band['max_legibility']} (hero shot)")
        if band.get("max_draws_attention") is not None and att > band["max_draws_attention"]:
            why.append(f"draws_attention {att} > {band['max_draws_attention']}")
        if band.get("front_attention") is not None and att >= band["front_attention"] \
                and str(inf.get("depth_plane", "")).lower() == "foreground":
            why.append(f"in the foreground and draws_attention {att} >= {band['front_attention']} (put right in front)")
        if band.get("gate_blind", True) and susp > band["max_blind_suspicion"]:
            why.append(f"blind viewer ad_suspicion {susp} > {band['max_blind_suspicion']}")
        if why:
            add("TOO_PROMINENT", "; ".join(why))
        if m["count"] > band["max_count"]:
            add("DUPLICATES", f"{m['count']} instances (max {band['max_count']})")
        # (unbranded look-alikes are reported but not judged: a diner full of signs is the scene, not a display)
        # "Inside the subject box" only means "on the subject" when the subject is a
        # compact object (a jacket, a face), not a whole group scene spanning the frame.
        compact = (m.get("subject_area") or 1) < 0.25
        if strategy != "incidental_use" and (m["subject_iou"] > 0.3 or (compact and m["inside_subject"] > 0.8)):
            add("ON_SUBJECT", f"product sits on the focal subject (IoU {m['subject_iou']}, "
                              f"{m['inside_subject']:.0%} inside subject box)")
        if inf.get("wordmark_correct") is False:
            add("GARBLED_BRAND", "brand text is misspelled/garbled")

    # -- scene preservation ----------------------------------------------
    lost = [x for x in inf.get("missing_or_altered") or [] if str(x).strip()]
    if fid < min_fidelity:
        add("FIDELITY_LOSS", f"fidelity {fid} < {min_fidelity}" + (f"; lost: {lost}" if lost else ""))
    elif len(lost) >= 2:
        add("FIDELITY_LOSS", f"lost/altered: {lost}")
    # Props the same-seed baseline had and this image lost (the product tends to *replace* clutter:
    # the office desk loses its lamp, mug and papers). Needs the two-image inspector call.
    ref = [x for x in inf.get("reference_objects") or [] if str(x).strip()]
    gone = [x for x in inf.get("missing_vs_reference") or [] if str(x).strip()]
    if len(ref) >= 2 and len(gone) >= 2 and len(gone) / len(ref) >= max_scene_loss:
        add("SCENE_LOSS", f"{len(gone)}/{len(ref)} baseline props gone: {gone}")
    if drift is not None and drift > max_drift:
        add("FIDELITY_LOSS", f"edit changed {drift:.1%} of the scene outside the product (max {max_drift:.0%})")
    if m["count"] and integ < min_integration:
        add("BAD_INTEGRATION", f"integration {integ} < {min_integration}")
    if int(inf.get("artifact_severity") or 0) >= 2:
        add("ARTIFACTS", f"severity {inf.get('artifact_severity')}: {inf.get('artifacts')}")

    # merge duplicate codes (e.g. two FIDELITY_LOSS reasons)
    merged: dict[str, dict] = {}
    for i in issues:
        if i["code"] in merged:
            merged[i["code"]]["detail"] += "; " + i["detail"]
        else:
            merged[i["code"]] = dict(i)
    return sorted(merged.values(), key=lambda i: PRIORITY.index(i["code"]))


def severity(issues: list[dict]) -> float:
    """Lower is better; used to pick the best attempt when none is accepted."""
    w = {c: len(PRIORITY) - i for i, c in enumerate(PRIORITY)}
    return sum(w[i["code"]] for i in issues)


def verify(llm, model: str, image: bytes, *, original_prompt: str, rewritten_prompt: str,
           brand: dict, plan_chosen: dict | None, band: dict, min_fidelity: int,
           min_integration: int, pre_edit_image: bytes | None = None, pre_edit_box: list | None = None,
           exclude_boxes: list | None = None, blind: bool = True,
           baseline_image: bytes | None = None, max_scene_loss: float = 0.5) -> dict:
    blind_out = blind_pass(llm, model, image, original_prompt) if blind else None
    ctx = {
        "original_prompt": original_prompt,
        "prompt_sent_to_image_model": rewritten_prompt,
        "intended_product": {"brand": brand["display"],
                             "form": (plan_chosen or {}).get("form"),
                             "anchor": (plan_chosen or {}).get("anchor"),
                             "spatial_relation": (plan_chosen or {}).get("spatial_relation"),
                             "frame_region": (plan_chosen or {}).get("frame_region"),
                             "orientation": (plan_chosen or {}).get("orientation")},
    }
    inf = informed_pass(llm, model, image, ctx, reference=baseline_image)
    m = box_metrics(inf.get("product_boxes"), inf.get("main_subject_box"), exclude_boxes)
    if exclude_boxes and m["count"] == 0 and inf.get("product_found"):
        inf = {**inf, "product_found": False}      # only the pre-existing instances were seen
    drift = None
    if pre_edit_image is not None:
        # mask both the new product box and wherever the product was before the edit
        boxes = [b for b in (m["largest_box"], pre_edit_box) if b]
        drift = edit_drift(pre_edit_image, image, boxes)
    issues = judge(inf, blind_out, m, band, min_fidelity, min_integration,
                   strategy=(plan_chosen or {}).get("strategy", ""), drift=drift,
                   max_scene_loss=max_scene_loss)
    return {"blind": blind_out, "informed": inf, "metrics": {**m, "edit_drift": drift},
            "brand_seen_blind": _brand_in(brand["display"], (blind_out or {}).get("brands_seen")),
            "issues": issues, "accepted": not issues}
