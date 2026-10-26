"""Builds docs/survey_set.html (+ docs/survey_img/ thumbnails) from runs/survey-closed.

    .venv/bin/python docs/build_survey_page.py
"""
import html
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "docs"
OUT = ROOT / "runs" / "survey-closed"
E = html.escape
sel = [s for s in json.loads((OUT / "selection.json").read_text()) if s["complete"]]
bench = json.loads((OUT / "bench.json").read_text())
seed = json.loads((ROOT / "runs/closed-seed-test/report.json").read_text())
(D / "survey_img").mkdir(exist_ok=True)


def thumb(src: Path, name: str) -> str:
    im = Image.open(src).convert("RGB")
    im.thumbnail((520, 520))
    im.save(D / "survey_img" / name, "JPEG", quality=78, optimize=True)
    return f"survey_img/{name}"


LABEL = {"base": "no-inject", "bad": "bad-inject", "good": "good-inject"}
SRC = {"v4f": "v4 rewriter · Gemini Flash", "v4s": "v4 rewriter · Claude Sonnet 5.5", "v2": "v2 rewriter",
       "naive": "naive append", "loop1": "feedback loop, round 1", "loop2": "feedback loop, round 2"}

cards = []
for n, s in enumerate(sel, 1):
    cells = []
    for role in ("base", "bad", "good"):
        src = OUT / f"{s['idx']:03d}" / f"{s[role]}.png"
        rel = thumb(src, f"S{n:02d}_{role}.jpg")
        sc = s["scores"][role]
        stat = (f"GPT ad? {sc['judge_ad']}/5" + (f" · legib {sc['legibility']}/4" if sc.get("legibility") is not None else "")
                + f" · CLIP {sc['clip']:.3f}")
        sent = s["prompts"][role]
        added = sent[len(s["prompt"]):].strip(" ,") if sent.startswith(s["prompt"]) else sent
        how = {"base": "original prompt" + (f" (re-roll: {s[role]})" if s[role] != "base" else ""),
               "bad": "product first, in the foreground" + (" · <b>weak</b>" if not s["bad_strong"] else ""),
               "good": SRC.get(s[role], s[role])}[role]
        detail = "" if role == "base" else f'<span class="ins">{E(added if role == "good" else sent[:90])}</span>'
        pair = s.get("pair_" + role) if role != "base" else None
        ptxt = ""
        if pair:
            ptxt = (f'<span class="pair {"hit" if pair["detected"] else "miss"}">side-by-side judge: '
                    f'{"picked this one" if pair["detected"] else "did not pick it"} ({pair.get("which_has_ad")})</span>')
        cells.append(f'''<figure class="cell"><img src="{rel}" alt="{E(LABEL[role])} image for item S{n:02d}" loading="lazy">
<figcaption><span class="role {role}">{LABEL[role]}</span><span class="how">{how}</span>{detail}<span class="stat">{E(stat)}</span>{ptxt}</figcaption></figure>''')
    cards.append(f'''<article class="item" id="S{n:02d}"><header><b>S{n:02d}</b><span class="brand">{E(s["brand"])}</span>
<span class="up">“{E(s["prompt"])}”</span></header><div class="trio">{"".join(cells)}</div></article>''')

b = bench["bench"]
order = [("naive", "Naive append (no LLM)"), ("v2", "v2 rewriter, Gemini Flash (12 prompts)"),
         ("v4f", "v4 rewriter, Gemini Flash"), ("v4s", "v4 rewriter, Claude Sonnet 5.5"), ("bad", "Bad-inject template")]
rows = "".join(
    f'<tr><td>{E(lab)}</td><td class="num">{b[k]["n"]}</td><td class="num">{b[k]["accepted"]}/{b[k]["n"]}</td>'
    f'<td class="num">{b[k]["survey_good"]}/{b[k]["n"]}</td><td class="num">{b[k]["brand_seen"]}/{b[k]["n"]}</td>'
    f'<td class="num">{b[k]["judge_ad"]:.2f}</td><td class="num">{b[k]["adherence"]:.2f}</td><td class="num">{b[k]["quality"]:.2f}</td>'
    f'<td class="num">{b[k]["clip"]:.3f}</td></tr>' for k, lab in order)
rows = (f'<tr class="ref"><td>Baseline (no inject)</td><td class="num">{b["base"]["n"]}</td><td class="num">–</td><td class="num">–</td>'
        f'<td class="num">{b["base"]["brand_seen"]}/{b["base"]["n"]} any</td><td class="num">{b["base"]["judge_ad"]:.2f}</td>'
        f'<td class="num">{b["base"]["adherence"]:.2f}</td><td class="num">{b["base"]["quality"]:.2f}</td><td class="num">{b["base"]["clip"]:.3f}</td></tr>') + rows

# bar chart: survey-good rate per condition
bars, W, L, R, rowh = [], 520, 228, 66, 34
items = [(lab, b[k]["survey_good"] / b[k]["n"], f'{b[k]["survey_good"]}/{b[k]["n"]}') for k, lab in order[:4]]
items.append(("+ feedback loop on the 8 failures", 3 / 8, "3/8 rescued"))
H = 14 + rowh * len(items) + 30
X = lambda v: L + v * (W - L - R)
svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Share of prompts whose image met the survey good-inject test, by rewriter condition: naive 3 of 21, v2 4 of 12, v4 Gemini Flash 6 of 21, v4 Claude Sonnet 6 of 20; the feedback loop rescued 3 of the 8 prompts that still had none.">']
for v in (0, .25, .5):
    svg.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="8" y2="{H - 26}" class="grid"/><text x="{X(v):.1f}" y="{H - 10}" text-anchor="middle" class="tick">{int(v * 100)}%</text>')
for i, (lab, v, t) in enumerate(items):
    y = 14 + i * rowh
    svg.append(f'<text x="{L - 10}" y="{y + 14}" text-anchor="end" class="rowlab">{E(lab)}</text>')
    svg.append(f'<rect x="{X(0):.1f}" y="{y}" width="{max(2, X(v) - X(0)):.1f}" height="20" rx="3" class="bar{" loop" if "loop" in lab else ""}"><title>{E(lab)}: {t}</title></rect>')
    svg.append(f'<text x="{X(v) + 8:.1f}" y="{y + 14}" class="vlab">{t}</text>')
svg.append("</svg>")

seedrows = "".join(
    f'<tr><td>{E(m)}</td><td class="num">{r.get("repeat_AB", "–")}</td><td class="num">{r.get("placed_AC", "–")}</td>'
    f'<td class="num">${sum(x["cost"] for k, x in r.items() if isinstance(x, dict)) / 4:.3f}</td>'
    f'<td class="num">{sum(x["sec"] for k, x in r.items() if isinstance(x, dict)) / 4:.0f} s</td></tr>' for m, r in seed.items())

pg = [s for s in sel if s.get("pair_good")]
pb = [s for s in sel if s.get("pair_bad")]
det_g = sum(s["pair_good"]["detected"] for s in pg)
det_b = sum(s["pair_bad"]["detected"] for s in pb)
cost_rows = "".join(f'<tr><td>{E(c["task"])}</td><td>{E(c["model"])}</td><td class="num">${c["usd"]:.2f}</td></tr>' for c in bench["cost"])

page = (D / "survey_set_template.html").read_text()
for k, v in {"CARDS": "".join(cards), "BENCH_ROWS": rows, "BARCHART": "".join(svg), "SEED_ROWS": seedrows,
             "COST_ROWS": cost_rows, "N_ITEMS": str(len(sel)), "N_IMAGES": str(3 * len(sel)),
             "N_STRONG_BAD": str(sum(s["bad_strong"] for s in sel)), "DET_G": f"{det_g}/{len(pg)}",
             "DET_B": f"{det_b}/{len(pb)}", "SESSION_USD": f"${bench['session_usd']:.2f}",
             "GOOD_AD": f"{sum(s['scores']['good']['judge_ad'] for s in sel) / len(sel):.1f}",
             "N_HFSS": str(sum(s["brand"] in ("Coca-Cola", "Pepsi", "Red Bull") for s in sel)),
             "PREF_G": f"{sum(s['pair_good']['prefers_injected'] for s in pg)}/{len(pg)}"}.items():
    page = page.replace("{{" + k + "}}", v)
(D / "survey_set.html").write_text(page)
print("built", len(sel), "items")
