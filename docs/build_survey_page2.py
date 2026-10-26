"""Builds docs/survey_set.html (v2: no-inject / simple / surface) from runs/survey-surface.

    .venv/bin/python docs/build_survey_page2.py
"""
import csv
import html
import json
import re
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "docs"
RUN = ROOT / "runs" / "survey-surface"
E = html.escape
sys.path.insert(0, str(ROOT))
from adplace.pipeline import load_catalog  # noqa: E402

CAT = load_catalog()
S = json.loads((RUN / "summary.json").read_text())
key = list(csv.DictReader((ROOT / "survey" / "key.csv").open()))
style = re.search(r"<style>.*?</style>", (D / "survey_set_template.html").read_text(), re.S).group(0)
style = style.replace(".role.bad{", ".role.simple{background:var(--accent-bg);color:var(--accent)}.role.bad{") \
             .replace(".role.good{", ".role.surface{background:var(--good-bg);color:var(--good)}.role.good{")
(D / "survey_img2").mkdir(exist_ok=True)
cost_led = [json.loads(l) for l in open(ROOT / "runs/ledger.jsonl")]
spent = sum(x["cost"] for x in cost_led) - 20.52


def thumb(src, name):
    im = Image.open(src).convert("RGB")
    im.thumbnail((520, 520))
    im.save(D / "survey_img2" / name, "JPEG", quality=78, optimize=True)
    return f"survey_img2/{name}"


def state(src):
    i = int(Path(src).parent.name)
    return i, json.loads((RUN / f"{i:03d}" / "state.json").read_text())


# ---------------------------------------------------------------- item cards
items = {}
for r in key:
    items.setdefault(r["item"], []).append(r)
cards = []
for it, rows in items.items():
    rows = sorted(rows, key=lambda r: ["no-inject", "simple", "surface"].index(r["condition"]))
    i, st = state(rows[0]["source"])
    cells = []
    for r in rows:
        rel = thumb(ROOT / r["source"], f"{it}_{r['condition']}.jpg")
        added = r["prompt_sent_to_model"][len(r["user_prompt"]):].strip(" ,")
        stem = Path(r["source"]).stem
        loop = f" (feedback round {stem[4:]})" if stem.startswith("loop") else ""
        pair = st.get("pair_" + r["condition"]) if r["condition"] != "no-inject" else None
        ptxt = (f'<span class="pair {"hit" if pair["detected"] else "miss"}">side-by-side judge: '
                f'{"picked this one" if pair["detected"] else "did not pick it"}</span>') if pair and pair.get("image") in (stem, None) else ""
        stat = f"GPT ad? {r['judge_ad_likelihood']}/5" + (f" · legib {r['legibility']}/4" if r["legibility"] else "") + \
               f" · CLIP {float(r['clipscore']):.3f}" + ("" if r["condition"] == "no-inject" else
                                                          (" · passes rule" if r["passes_rule"] == "True" else " · fails rule"))
        cells.append(f'''<figure class="cell"><img src="{rel}" alt="{E(r['condition'])} image for {it}" loading="lazy">
<figcaption><span class="role {r['condition'].replace('no-inject', 'base')}">{E(r['condition'])}{E(loop)}</span>
{f'<span class="ins">+ “{E(added)}”</span>' if added else '<span class="how">original prompt</span>'}<span class="stat">{E(stat)}</span>{ptxt}</figcaption></figure>''')
    cards.append(f'''<article class="item" id="{it}"><header><b>{it}</b><span class="brand">{E(rows[1]["brand"])}</span>
<span class="up">“{E(rows[0]["user_prompt"])}”</span></header><div class="trio">{"".join(cells)}</div></article>''')

# ---------------------------------------------------------------- staging examples (simple vs surface)
ex = []
for i in (10, 21, 17, 23):
    st = json.loads((RUN / f"{i:03d}" / "state.json").read_text())
    cells = []
    for k in ("base", "simple", "surface"):
        c = st["cond"][k]
        rel = thumb(RUN / f"{i:03d}" / f"{k}.png", f"ex{i}_{k}.jpg")
        j = c.get("judge") or {}
        lab = {"base": "original", "simple": f'+ “with {st["phrase"]["product"]}”',
               "surface": f'+ “with {st["phrase"]["product"]} {st["phrase"]["surface"]}”'}[k]
        cells.append(f'<figure class="cell"><img src="{rel}" alt="{k} example" loading="lazy"><figcaption>'
                     f'<span class="ins">{E(lab)}</span><span class="stat">GPT ad? {j.get("ad_likelihood")}/5 · '
                     f'natural {j.get("brand_naturalness") or "–"}/5</span></figcaption></figure>')
    ex.append(f'<div class="exrow"><span class="up">“{E(st["prompt"])}”</span><div class="trio">{"".join(cells)}</div></div>')

# ---------------------------------------------------------------- benchmark table + bar chart
order = [("simple", "Simple: “…, with a can of Coke”"), ("surface", "Surface: “…, with a can of Coke on the cafe table”"),
         ("surface_short", "Surface, short reference: “…on their table”"), ("surface+loop", "Surface + feedback loop (≤2 new surfaces)")]
rows = (f'<tr class="ref"><td>No-inject (original prompt)</td><td class="num">{S["base"]["n"]}</td><td class="num">–</td>'
        f'<td class="num">{S["base"]["judge_ad"]:.2f}</td><td class="num">–</td><td class="num">–</td><td class="num">–</td>'
        f'<td class="num">{S["base"]["adherence"]:.2f}</td><td class="num">{S["base"]["clip"]:.3f}</td></tr>')
for k, lab in order:
    s = S[k]
    rows += (f'<tr><td>{E(lab)}</td><td class="num">{s["n"]}</td><td class="num">{s["accepted"]}/{s["n"]}</td>'
             f'<td class="num">{s["judge_ad"]:.2f}</td><td class="num">{s["naturalness"]:.2f}</td>'
             f'<td class="num">{s["legibility"]:.2f}</td><td class="num">{s.get("pair_detected") or "–"}</td>'
             f'<td class="num">{s["adherence"]:.2f}</td><td class="num">{s["clip"]:.3f}</td></tr>')
W, L, R, rowh = 560, 250, 60, 34
H = 14 + rowh * len(order) + 30
X = lambda v: L + v * (W - L - R)
svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Share of 24 prompts whose image passes the acceptance rule: simple 16, surface 15, surface short 11, surface plus feedback loop 20.">']
for v in (0, .5, 1):
    svg.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="8" y2="{H - 26}" class="grid"/><text x="{X(v):.1f}" y="{H - 10}" text-anchor="middle" class="tick">{int(v * 100)}%</text>')
for n, (k, lab) in enumerate(order):
    s = S[k]
    v = s["accepted"] / s["n"]
    y = 14 + n * rowh
    short = {"simple": "Simple “with X”", "surface": "Surface “with X on Y”", "surface_short": "Surface, short reference",
             "surface+loop": "Surface + feedback loop"}[k]
    svg.append(f'<text x="{L - 10}" y="{y + 14}" text-anchor="end" class="rowlab">{E(short)}</text>')
    svg.append(f'<rect x="{X(0):.1f}" y="{y}" width="{X(v) - X(0):.1f}" height="20" rx="3" class="bar{" loop" if "loop" in k else ""}"><title>{E(short)}: {s["accepted"]}/{s["n"]}</title></rect>')
    svg.append(f'<text x="{X(v) + 8:.1f}" y="{y + 14}" class="vlab">{s["accepted"]}/{s["n"]}</text>')
svg.append("</svg>")

ledger_sess = [x for x in cost_led if x.get("run_id") == "survey-surface"]
by = {}
for x in ledger_sess:
    k = (x["task"].split(":")[0], x["model"])
    by[k] = by.get(k, 0) + x["cost"]
cost_rows = "".join(f'<tr><td>{E(t)}</td><td>{E(m)}</td><td class="num">${v:.2f}</td></tr>'
                    for (t, m), v in sorted(by.items(), key=lambda kv: -kv[1]))
n_items = len(items)

page = f"""<title>Survey Image Set</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
{style}
<style>.exrow{{display:grid;gap:6px}}.exs{{display:grid;gap:18px}}</style>
<div class="wrap">
<header class="top">
  <span class="eyebrow">Thesis C · survey stimuli v2 · 4 Oct 2026</span>
  <h1>Survey image set: original vs simple vs surface-specific rewrite</h1>
  <p class="muted">{n_items} everyday prompts, each rendered three ways: the <b>original prompt</b>; a <b>simple</b> rewrite that just adds the product (“…, with a can of Coke”); and a <b>surface-specific</b> rewrite that also says where it sits (“…, with a can of Coke on the cafe table”). A single LLM step writes the product phrase and the surface, so the two rewrites differ only by the surface clause. All images are from Gemini 3.1 Flash-Lite Image, because Comfy Cloud/FLUX is unavailable.</p>
</header>

<div class="warn"><b>Researcher view.</b> Give participants only <code>survey/images/</code>. File names are S01_A…S{n_items:02d}_C, with letters shuffled per item. The answer key is <code>survey/key.csv</code>. This replaces the earlier good/bad-inject set, which is kept in <code>survey_v1_good-bad/</code>.</div>

<div class="kpis">
  <div class="kpi"><b>{n_items} × 3</b><span>items × images ({3 * n_items} total); every surface image passes your rule</span></div>
  <div class="kpi"><b>{S["simple"]["accepted"]} vs {S["surface"]["accepted"]}</b><span>of 24 pass your rule on the first try: simple vs surface</span></div>
  <div class="kpi"><b>{S["surface+loop"]["accepted"]}/24</b><span>surface-specific after the feedback loop picks a new surface (≤2 tries)</span></div>
  <div class="kpi"><b>{S["simple"]["naturalness"]:.1f} vs {S["surface"]["naturalness"]:.1f}</b><span>how naturally the brand fits the scene (GPT, 1–5): simple vs surface</span></div>
</div>

<section>
  <h2>How the rewrites are built</h2>
  <figure><div class="diagram">
  <svg viewBox="0 0 980 230" role="img" aria-label="One LLM call sees the original image and writes the product phrase and a surface; the simple rewrite appends 'with product', the surface rewrite appends 'with product surface'; both go to Gemini 3.1 Flash-Lite and are verified; a failed surface image goes to a feedback step that picks a different surface.">
    <defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="currentColor"/></marker>
    <marker id="aA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--accent)"/></marker></defs>
    <rect x="10" y="80" width="150" height="58" rx="6" class="box"/><text x="85" y="104" text-anchor="middle" class="ttl">Original prompt</text><text x="85" y="122" text-anchor="middle" class="sub">+ brand from classifier</text>
    <rect x="190" y="80" width="180" height="58" rx="6" class="box hi"/><text x="280" y="104" text-anchor="middle" class="ttl">Phrase LLM (sees image)</text><text x="280" y="122" text-anchor="middle" class="sub">product · surface · 2 alternatives</text>
    <line x1="160" y1="109" x2="188" y2="109" class="edge" marker-end="url(#a)"/>
    <rect x="410" y="20" width="230" height="50" rx="6" class="box"/><text x="525" y="41" text-anchor="middle" class="ttl">simple</text><text x="525" y="58" text-anchor="middle" class="sub">“…, with a can of Coke”</text>
    <rect x="410" y="148" width="230" height="50" rx="6" class="box hi"/><text x="525" y="169" text-anchor="middle" class="ttl">surface</text><text x="525" y="186" text-anchor="middle" class="sub">“…, with a can of Coke on the cafe table”</text>
    <path d="M370 109 H390 V45 H408" class="edge" marker-end="url(#a)"/><path d="M390 109 V173 H408" class="edge" marker-end="url(#a)"/>
    <rect x="680" y="84" width="140" height="50" rx="6" class="box"/><text x="750" y="105" text-anchor="middle" class="ttl">Gemini Flash-Lite</text><text x="750" y="122" text-anchor="middle" class="sub">image, 1:1, seed 7</text>
    <path d="M640 45 H660 V109 H678" class="edge" marker-end="url(#a)"/><path d="M640 173 H660 V109" class="edge"/>
    <rect x="850" y="84" width="120" height="50" rx="6" class="box hi"/><text x="910" y="105" text-anchor="middle" class="ttl">Verify + judge</text><text x="910" y="122" text-anchor="middle" class="sub">rule · GPT · CLIP</text>
    <line x1="820" y1="109" x2="848" y2="109" class="edge" marker-end="url(#a)"/>
    <path d="M910 134 V218 H525 V200" class="edge acc" marker-end="url(#aA)"/>
    <text x="718" y="212" text-anchor="middle" class="elab acc">surface fails → feedback picks a different surface (≤2 tries)</text>
  </svg></div>
  <figcaption class="cap">“Your rule” is the calibrated band fitted to your 160 lightbox labels: not blatant or right in front (no hero-shot legibility, no foreground product that grabs the eye), blind ad-suspicion ≤4, and the scene intact. A Gemini 3.7 Flash verifier applies it. GPT-5.6 Sol acts as an independent viewer, told nothing, and rates how ad-like and how natural each image is. CLIPScore is scored against the original prompt.</figcaption></figure>
</section>

<section>
  <h2>Simple vs surface-specific</h2>
  <div class="two">
    <figure><div class="chart">{"".join(svg)}</div><figcaption class="cap">Share of the 24 prompts whose image passes your rule. The orange bar includes the feedback loop.</figcaption></figure>
    <div style="display:grid;gap:10px">
      <p><b>On the first try, naming the surface did not beat the simple rewrite.</b> The surface version made the brand slightly more readable (legibility {S["surface"]["legibility"]:.2f} vs {S["simple"]["legibility"]:.2f}). But it also looked more staged: GPT rated how naturally the brand fits at {S["surface"]["naturalness"]:.2f} vs {S["simple"]["naturalness"]:.2f}. It failed as too prominent more often ({S["surface"]["too_prominent"]} vs {S["simple"]["too_prominent"]}), and a side-by-side viewer spotted it more often ({S["surface"]["pair_detected"]} vs {S["simple"]["pair_detected"]}).</p>
      <p><b>The surface condition wins once the feedback loop runs.</b> When a surface image fails, the loop looks at it and picks a different surface. That lifts surface-specific placement to {S["surface+loop"]["accepted"]}/24, ahead of simple's {S["simple"]["accepted"]}/24, while keeping control over where the product goes, which is what an advertiser pays for.</p>
      <p>Using a shorter reference to the surface (“on their table” instead of “on the round bistro table”) didn't help ({S["surface_short"]["accepted"]}/24). The problem is naming a surface at all, not how it's described.</p>
    </div>
  </div>
  <div class="tablewrap"><table>
    <thead><tr><th>Condition</th><th class="num">n</th><th class="num">Passes rule</th><th class="num">GPT ad? (1–5)</th><th class="num">Natural fit (1–5)</th><th class="num">Legibility (0–4)</th><th class="num">Side-by-side spotted</th><th class="num">Adherence</th><th class="num">CLIPScore</th></tr></thead>
    <tbody>{rows}</tbody></table></div>
  <p class="muted">Same 24 prompts, image model and seed for every condition. The loop row reports the image each prompt ended on. Side-by-side: a GPT judge saw the original and the rewritten image in random order and was asked which, if either, contains advertising.</p>
</section>

<section>
  <h2>Why the surface version looks staged</h2>
  <p>A surface named in the prompt becomes something the model wants to show. It re-frames the shot to bring the towel, the table or the ground into the foreground, with the product sitting on it. With just “with a can of Coke”, the product usually ends up somewhere already in the scene: in a hand, among the food, in a cup holder. The cup-holder example (bottom row) is the kind of surface that works, because it's somewhere the camera already looks.</p>
  <div class="exs">{"".join(ex)}</div>
</section>

<section>
  <h2>A note for the survey design</h2>
  <p>Shown side by side with the original, the GPT viewer picked out the rewritten image as the one with advertising in {S["simple"]["pair_detected"]} pairs for simple and {S["surface"]["pair_detected"]} for surface. Seen alone, the same images averaged only about 3/5 on “looks like an ad”. If participants see both versions of a prompt together, the extra branded object is the obvious difference. Consider showing the two versions in separate blocks or with filler items between them, or asking about each image before showing its pair.</p>
</section>

<section>
  <h2>The {n_items} items</h2>
  <p class="muted">The surface image is the one the feedback loop ended on, so every surface image here passes your rule. Simple images are shown as generated, whether or not they pass. The loop picked a new surface for some items; the label says which round.</p>
  <div class="items">{"".join(cards)}</div>
</section>

<section>
  <h2>Notes and caveats</h2>
  <ul class="tight">
    <li><b>Separate renders.</b> Closed image models redraw the whole scene when the prompt changes, even on a fixed seed. The three images of an item are different compositions of the same prompt, not identical except for the product (FLUX's same-seed pairing needs Comfy Cloud back).</li>
    <li><b>Dropped items.</b> 4 of the 24 prompts never got a surface image that passed your rule, even after the loop, so they aren't in the set.</li>
    <li><b>Re-rolls.</b> Flash-Lite occasionally letterboxes an image with white bars. The two affected simple images (Nike basketball, Starbucks office kitchen) were re-rolled on another seed. The original images reuse brand-free seeds from the earlier run.</li>
    <li><b>Organic brands</b> that belong to a scene (the Lawson store sign, Liverpool FC at the stadium) appear in some original images.</li>
    <li><b>Judges are models.</b> The verifier uses your fitted rule (72% agreement with your labels). The GPT viewer is a proxy for participants.</li>
  </ul>
  <div class="tablewrap"><table><thead><tr><th>Spend (simple vs surface run)</th><th>Model</th><th class="num">USD</th></tr></thead>
  <tbody>{cost_rows}<tr><td><b>Total, incl. letterbox re-rolls</b> (budget $6)</td><td></td><td class="num"><b>${spent:.2f}</b></td></tr></tbody></table></div>
</section>
</div>
"""
(D / "survey_set.html").write_text(page)
print("built", n_items, "items; spend", round(spent, 2))
