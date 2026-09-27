"""Builds the SVG charts for research_brief.html from pilot data and splices them into the page.

    .venv/bin/python docs/build_brief_charts.py
Reads docs/brief_points.json (every prompt-only verified image in runs A, B, G) and docs/brief_clip.json.
"""
import html
import json
import math
import random
from pathlib import Path

D = Path(__file__).parent
pts = json.loads((D / "brief_points.json").read_text())
clip = json.loads((D / "brief_clip.json").read_text())
NAMES = {"0": "picnic (painting)", "1": "firefighter BBQ", "2": "1950s diner", "3": "office desk",
         "4": "road trip", "5": "dorm study", "6": "rooftop bar", "16": "dystopian city"}


def scatter() -> str:
    W, H, L, R, T, B = 720, 360, 56, 20, 16, 48
    xmin, xmax = math.log10(0.001), math.log10(0.2)          # 0.1% .. 20% of frame
    X = lambda a: L + (math.log10(max(a, 0.001)) - xmin) / (xmax - xmin) * (W - L - R)
    Y = lambda s: T + (5.5 - s) / 5 * (H - T - B)              # 1..5 with half-step padding
    rnd = random.Random(7)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Scatter of product area against blind-viewer ad suspicion for 87 prompt-only images. Almost every point sits above the subtle band ceiling of 2; v2 points sit further left (smaller) than v1 points.">']
    # subtle band: area 0.2%..4%, suspicion <= 2
    x0, x1, y0, y1 = X(0.002), X(0.04), Y(2.5), Y(0.5)
    out.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0 - 0:.1f}" class="band"/>')
    out.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{y0 + 16:.1f}" text-anchor="middle" class="bandlab">subtle band (target)</text>')
    # grid + axes
    for s in range(1, 6):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(s):.1f}" y2="{Y(s):.1f}" class="grid"/>')
        out.append(f'<text x="{L - 10}" y="{Y(s) + 4:.1f}" text-anchor="end" class="tick">{s}</text>')
    for a, lab in [(0.001, "0.1%"), (0.002, "0.2%"), (0.005, "0.5%"), (0.01, "1%"), (0.02, "2%"), (0.04, "4%"), (0.1, "10%"), (0.2, "20%")]:
        out.append(f'<line x1="{X(a):.1f}" x2="{X(a):.1f}" y1="{T}" y2="{H - B}" class="grid"/>')
        out.append(f'<text x="{X(a):.1f}" y="{H - B + 18}" text-anchor="middle" class="tick">{lab}</text>')
    out.append(f'<text x="{(L + W - R) / 2}" y="{H - 8}" text-anchor="middle" class="axlab">product area, share of frame (log scale)</text>')
    out.append(f'<text transform="translate(14 {(T + H - B) / 2}) rotate(-90)" text-anchor="middle" class="axlab">blind viewer: “looks like an ad” (1–5)</text>')
    for p in sorted(pts, key=lambda p: p["ver"]):
        if not p["susp"]:
            continue
        cx = X(p["area"]) + rnd.uniform(-2, 2)
        cy = Y(p["susp"]) + rnd.uniform(-11, 11)
        cls = "v1" if p["ver"] == "v1" else "v2"
        tip = (f'{NAMES.get(str(p["idx"]), p["idx"])} · {"v1" if cls == "v1" else "v2"} attempt {p["k"]}\n'
               f'area {p["area"] * 100:.2f}% · legibility {p["leg"]}/4 · ad suspicion {p["susp"]}/5'
               + (" · ACCEPTED" if p["ok"] else ""))
        r = 6 if p["ok"] else 4.5
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" class="pt {cls}{" ok" if p["ok"] else ""}"><title>{html.escape(tip)}</title></circle>')
    out.append("</svg>")
    return "".join(out)


def dumbbell() -> str:
    rows = sorted(clip.items(), key=lambda kv: kv[1]["base"] - kv[1]["pilot-edit"], reverse=True)
    W, rowh, L, R, T = 720, 34, 150, 24, 30
    H = T + rowh * len(rows) + 40
    lo, hi = 0.22, 0.39
    X = lambda v: L + (v - lo) / (hi - lo) * (W - L - R)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="CLIPScore against the original prompt for each pilot prompt: baseline, v1 first attempt and v2 first attempt. v2 stays closer to the baseline in 5 of 8 prompts; mean change −0.028 for v1 and −0.010 for v2.">']
    for v in (0.24, 0.28, 0.32, 0.36):
        out.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{T - 10}" y2="{H - 34}" class="grid"/>')
        out.append(f'<text x="{X(v):.1f}" y="{H - 16}" text-anchor="middle" class="tick">{v:.2f}</text>')
    for i, (idx, c) in enumerate(rows):
        y = T + i * rowh + rowh / 2
        b, v1, v2 = c["base"], c["pilot-edit"], c["pilot-v2-regen"]
        out.append(f'<text x="{L - 12}" y="{y + 4:.1f}" text-anchor="end" class="rowlab">{html.escape(NAMES[idx])}</text>')
        out.append(f'<line x1="{X(min(b, v1, v2)):.1f}" x2="{X(max(b, v1, v2)):.1f}" y1="{y}" y2="{y}" class="stem"/>')
        for val, cls, lab in ((b, "base", "baseline"), (v1, "v1", "v1 rewrite"), (v2, "v2", "v2 rewrite")):
            out.append(f'<circle cx="{X(val):.1f}" cy="{y}" r="6" class="pt {cls}"><title>{html.escape(NAMES[idx])} · {lab}: {val:.3f} (Δ {val - b:+.3f} vs baseline)</title></circle>')
    out.append("</svg>")
    return "".join(out)


page = (D / "research_brief.html").read_text()
for tag, svg in (("SCATTER", scatter()), ("DUMBBELL", dumbbell())):
    start, end = f"<!--{tag}-->", f"<!--/{tag}-->"
    a, b = page.index(start) + len(start), page.index(end)
    page = page[:a] + svg + page[b:]
(D / "research_brief.html").write_text(page)
print("charts spliced")
