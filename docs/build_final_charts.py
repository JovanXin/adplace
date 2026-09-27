"""Splices data-driven SVG charts and tables into docs/final_report.html.

    .venv/bin/python docs/final_stats.py        # recompute data
    .venv/bin/python docs/build_final_charts.py # rebuild charts in the page
"""
import html
import json
import math
import random
from pathlib import Path

D = Path(__file__).parent
S = json.loads((D / "final_stats.json").read_text())
P = json.loads((D / "final_points.json").read_text())
NAMES = {0: "picnic (painting)", 1: "firefighter BBQ", 2: "1950s diner", 3: "office desk", 4: "road trip",
         5: "dorm study", 6: "rooftop bar", 16: "dystopian city"}
ORDER = ["A", "B", "G", "C", "D", "E", "F"]
SHORT = {"A": "A · v1 + Qwen edit", "B": "B · v1 + prompt repair", "G": "G · v2 + prompt repair",
         "C": "C · v2 + Qwen edit", "D": "D · VPP, Qwen insert", "E": "E · VPP, Gemini insert", "F": "F · Gemini gen + edit"}
E = html.escape


def family(p):
    if p["run"] == "F":
        return "gemini"
    if p["how"] == "edit":
        return "gemini" if p["run"] == "E" else "qwen"
    return "flux"


FAM = {"flux": ("s1", "FLUX.2 dev, prompt only"), "gemini": ("s2", "Gemini 3.1 Flash Image (closed)"),
       "qwen": ("s3", "Qwen-Image-Edit (open)")}


def scatter():
    W, H, L, R, T, B = 760, 380, 56, 20, 16, 48
    xmin, xmax = math.log10(0.001), math.log10(0.2)
    X = lambda a: L + (math.log10(max(a, 0.001)) - xmin) / (xmax - xmin) * (W - L - R)
    Y = lambda s: T + (5.5 - s) / 5 * (H - T - B)
    rnd = random.Random(11)
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Scatter of {len(P)} verified images: product area against blind ad suspicion, coloured by the model that drew the product. Gemini points sit lower at the same size; FLUX prompt-only points cluster at suspicion 4 to 5.">']
    x0, x1, y0, y1 = X(0.002), X(0.04), Y(2.5), Y(0.5)
    o.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" class="band"/>')
    o.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{y1 - 10:.1f}" text-anchor="middle" class="bandlab">subtle band</text>')
    for s in range(1, 6):
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(s):.1f}" y2="{Y(s):.1f}" class="grid"/><text x="{L - 10}" y="{Y(s) + 4:.1f}" text-anchor="end" class="tick">{s}</text>')
    for a, lab in [(0.001, "0.1%"), (0.002, "0.2%"), (0.005, "0.5%"), (0.01, "1%"), (0.02, "2%"), (0.04, "4%"), (0.1, "10%"), (0.2, "20%")]:
        o.append(f'<line x1="{X(a):.1f}" x2="{X(a):.1f}" y1="{T}" y2="{H - B}" class="grid"/><text x="{X(a):.1f}" y="{H - B + 18}" text-anchor="middle" class="tick">{lab}</text>')
    o.append(f'<text x="{(L + W - R) / 2}" y="{H - 8}" text-anchor="middle" class="axlab">product area, share of frame (log scale; 0.1% = not found)</text>')
    o.append(f'<text transform="translate(14 {(T + H - B) / 2}) rotate(-90)" text-anchor="middle" class="axlab">blind viewer: “looks like an ad” (1–5)</text>')
    for p in sorted(P, key=lambda p: {"flux": 0, "qwen": 1, "gemini": 2}[family(p)]):
        if not p["susp"]:
            continue
        cls, lab = FAM[family(p)]
        tip = f'Run {p["run"]} · {NAMES.get(p["idx"], p["idx"])} · attempt {p["k"]} ({p["how"]})\n{lab}\narea {p["area"] * 100:.2f}% · legibility {p["leg"]}/4 · ad suspicion {p["susp"]}/5' + (" · ACCEPTED" if p["ok"] else "")
        o.append(f'<circle cx="{X(p["area"]) + rnd.uniform(-2, 2):.1f}" cy="{Y(p["susp"]) + rnd.uniform(-11, 11):.1f}" r="{6 if p["ok"] else 4.5}" class="pt {cls}{" ok" if p["ok"] else ""}"><title>{E(tip)}</title></circle>')
    o.append("</svg>")
    return "".join(o)


def legibility_curve():
    W, H, L, R, T, B = 560, 300, 50, 110, 20, 46
    X = lambda l: L + (l - 1) / 3 * (W - L - R)
    Y = lambda s: T + (5 - s) / 4 * (H - T - B)
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Mean blind ad suspicion by legibility level. FLUX rises from 2.0 to 4.6; Gemini stays lower, 2.2 to 3.4. The band ceiling of 2 is crossed at legibility 2 for both.">']
    for s in range(1, 6):
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(s):.1f}" y2="{Y(s):.1f}" class="grid"/><text x="{L - 10}" y="{Y(s) + 4:.1f}" text-anchor="end" class="tick">{s}</text>')
    o.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(2):.1f}" y2="{Y(2):.1f}" class="ceil"/><text x="{W - R + 6}" y="{Y(2) + 4:.1f}" class="ceillab">subtle ceiling</text>')
    for l, lab in [(1, "1 shape/colour"), (2, "2 partial mark"), (3, "3 readable"), (4, "4 hero")]:
        o.append(f'<text x="{X(l):.1f}" y="{H - B + 18}" text-anchor="middle" class="tick">{lab}</text>')
    o.append(f'<text x="{(L + W - R) / 2}" y="{H - 6}" text-anchor="middle" class="axlab">legibility of the brand (inspector, 0–4)</text>')
    for fam in ("flux", "gemini"):
        cls, lab = FAM[fam]
        pts = []
        for l in range(1, 5):
            s = [p["susp"] for p in P if family(p) == fam and p["leg"] == l and p["susp"]]
            if s:
                pts.append((l, sum(s) / len(s), len(s)))
        o.append('<polyline fill="none" class="ln {}" points="{}"/>'.format(cls, " ".join(f"{X(l):.1f},{Y(m):.1f}" for l, m, _ in pts)))
        for l, m, n in pts:
            o.append(f'<circle cx="{X(l):.1f}" cy="{Y(m):.1f}" r="5" class="pt {cls}"><title>{E(lab)} · legibility {l}: mean suspicion {m:.2f} (n={n})</title></circle>')
        l, m, _ = pts[-1]
        o.append(f'<text x="{X(l) + 10:.1f}" y="{Y(m) + 4:.1f}" class="dlab">{"FLUX" if fam == "flux" else "Gemini"}</text>')
    o.append("</svg>")
    return "".join(o)


def cost_bars():
    rows = [(k, S[k]) for k in ORDER]
    W, rowh, L, R, T = 620, 30, 190, 70, 10
    H = T + rowh * len(rows) + 34
    mx = 0.40
    X = lambda v: L + v / mx * (W - L - R)
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="OpenRouter cost per injected prompt by condition. Open-model conditions cost 4 to 11 cents, mostly LLM calls; the closed-image conditions cost about 35 cents, mostly image generation and editing.">']
    for v in (0, 0.1, 0.2, 0.3, 0.4):
        o.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{T}" y2="{H - 28}" class="grid"/><text x="{X(v):.1f}" y="{H - 12}" text-anchor="middle" class="tick">${v:.2f}</text>')
    for i, (k, s) in enumerate(rows):
        y = T + i * rowh + 6
        llm = (s["usd"] - s["usd_img"]) / max(1, s["n"])
        img = s["usd_img"] / max(1, s["n"])
        o.append(f'<text x="{L - 10}" y="{y + 13}" text-anchor="end" class="rowlab">{E(SHORT[k])}</text>')
        o.append(f'<rect x="{X(0):.1f}" y="{y}" width="{max(0, X(llm) - X(0)):.1f}" height="18" rx="2" class="bar s1"><title>Run {k}: LLM calls ${llm:.3f}/prompt</title></rect>')
        if img:
            o.append(f'<rect x="{X(llm) + 2:.1f}" y="{y}" width="{max(0, X(llm + img) - X(llm) - 2):.1f}" height="18" rx="2" class="bar s2"><title>Run {k}: closed image model ${img:.3f}/prompt</title></rect>')
        o.append(f'<text x="{X(llm + img) + 6:.1f}" y="{y + 13}" class="vlab">${llm + img:.3f}</text>')
    o.append("</svg>")
    return "".join(o)


def clip_dots():
    rows = [(k, S[k]) for k in ORDER if S[k]["clip"]]
    W, rowh, L, R, T = 620, 30, 190, 60, 10
    H = T + rowh * len(rows) + 34
    lo, hi = -0.04, 0.01
    X = lambda v: L + (v - lo) / (hi - lo) * (W - L - R)
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Mean CLIPScore change of the first attempt versus the same-seed baseline, by condition. v1 conditions lose about 0.028; v2 loses 0.010; VPP conditions are near zero.">']
    for v in (-0.04, -0.03, -0.02, -0.01, 0, 0.01):
        o.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{T}" y2="{H - 28}" class="{"zero" if v == 0 else "grid"}"/><text x="{X(v):.1f}" y="{H - 12}" text-anchor="middle" class="tick">{v:+.2f}</text>')
    for i, (k, s) in enumerate(rows):
        y = T + i * rowh + 15
        v = s["clip_d_att1"]
        o.append(f'<text x="{L - 10}" y="{y + 4}" text-anchor="end" class="rowlab">{E(SHORT[k])} (n={len(s["clip"])})</text>')
        o.append(f'<line x1="{X(0):.1f}" x2="{X(v):.1f}" y1="{y}" y2="{y}" class="stem"/>')
        o.append(f'<circle cx="{X(v):.1f}" cy="{y}" r="6" class="pt s1"><title>Run {k}: mean Δ CLIPScore attempt 1 {v:+.4f}; final {s["clip_d_final"]:+.4f}</title></circle>')
        o.append(f'<text x="{X(v) + (10 if v >= 0 else -10):.1f}" y="{y + 4}" text-anchor="{"start" if v >= 0 else "end"}" class="vlab">{v:+.3f}</text>')
    o.append("</svg>")
    return "".join(o)


def accept_table():
    cols = [("subtle", "Subtle"), ("subtle_noblind", "Subtle, no blind gate"), ("moderate", "Moderate"), ("moderate_noblind", "Moderate, no blind gate")]
    h = ['<div class="tablewrap"><table class="heat"><thead><tr><th>Condition</th><th class="num">n</th>'
         + "".join(f'<th class="num">{E(c)}</th>' for _, c in cols) + '<th class="num">Mean attempts</th></tr></thead><tbody>']
    for k in ORDER:
        s = S[k]
        cells = ""
        for key, _ in cols:
            v = s["variants"][key]
            pct = round(100 * v / s["n"]) if s["n"] else 0
            cells += f'<td class="num" style="--p:{pct}%">{v}/{s["n"]}</td>'
        h.append(f'<tr><td title="{E(s["label"])}" class="cond">{E(SHORT[k])}</td><td class="num">{s["n"]}</td>{cells}<td class="num">{s["mean_attempts"]}</td></tr>')
    h.append("</tbody></table></div>")
    return "".join(h)


def stats_table():
    h = ['<div class="tablewrap"><table><thead><tr><th>Condition</th><th class="num">Median area</th><th class="num">Mean legibility</th>'
         '<th class="num">Mean ad-suspicion</th><th class="num">Brand named by blind viewer</th><th class="num">Mean edit drift</th><th class="num">Top issue</th></tr></thead><tbody>']
    for k in ORDER:
        s = S[k]
        top = max(s["codes"].items(), key=lambda kv: kv[1]) if s["codes"] else ("–", 0)
        tot = sum(s["codes"].values())
        h.append(f'<tr><td title="{E(s["label"])}" class="cond">{E(SHORT[k])}</td><td class="num">{s["median_area"] * 100:.1f}%</td><td class="num">{s["mean_leg"]:.2f}</td>'
                 f'<td class="num">{s["mean_susp"]:.2f}</td><td class="num">{s["brand_seen_pct"]:.0f}%</td>'
                 f'<td class="num">{("%.1f%%" % (s["mean_drift"] * 100)) if s["mean_drift"] is not None else "–"}</td>'
                 f'<td class="num">{E(top[0])} {round(100 * top[1] / tot)}%</td></tr>')
    h.append("</tbody></table></div>")
    return "".join(h)


page = (D / "final_report.html").read_text()
for tag, content in (("SCATTER", scatter()), ("LEGCURVE", legibility_curve()), ("COST", cost_bars()),
                     ("CLIP", clip_dots()), ("ACCEPT", accept_table()), ("STATS", stats_table())):
    a, b = page.index(f"<!--{tag}-->") + len(f"<!--{tag}-->"), page.index(f"<!--/{tag}-->")
    page = page[:a] + content + page[b:]
(D / "final_report.html").write_text(page)
print("spliced")
