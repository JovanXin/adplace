"""Static HTML report for one or more runs: summary, issue histogram, and a contact sheet per prompt.

    python -m adplace report runs/<run_id>
    build([run_a, run_b], out_dir)  -> one page comparing runs (e.g. edit vs regen repair)
Images are written as downscaled JPEGs next to the page (out_dir/img/...).
"""
from __future__ import annotations

import difflib
import html
import io
import json
import re
from pathlib import Path

from .verify import PRIORITY

E = html.escape


def _load(run_dir: Path) -> tuple[dict, list[dict], dict]:
    recs = [json.loads(p.read_text()) for p in sorted(run_dir.glob("[0-9][0-9][0-9]/record.json"))]
    summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    cfg = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
    return summary, recs, cfg


def _thumb(src: Path, dst: Path, width: int = 640) -> str:
    from PIL import Image
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        im = Image.open(src).convert("RGB")
        im.thumbnail((width, width * 2))
        im.save(dst, format="JPEG", quality=82, optimize=True)
    return dst.as_posix()


def _diff_html(original: str, new: str) -> str:
    a, b = original.split(), new.split()
    norm = lambda ws: [re.sub(r"[^\w']", "", w.lower()) for w in ws]
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=norm(a), b=norm(b), autojunk=False).get_opcodes():
        seg = E(" ".join(b[j1:j2]))
        if op == "equal":
            out.append(seg)
        elif op in ("insert", "replace") and seg:
            out.append(f"<mark>{seg}</mark>")
        if op in ("delete", "replace") and i2 > i1:
            out.append(f"<del>{E(' '.join(a[i1:i2]))}</del>")
    return " ".join(x for x in out if x)


def _pill(text: str, kind: str) -> str:
    return f'<span class="pill {kind}">{E(text)}</span>'


STATUS_KIND = {"accepted": "ok", "failed": "bad", "error": "bad", "budget_stopped": "bad",
               "skip": "muted", "exclude": "muted", "dry_run": "muted"}


def _metrics_line(v: dict) -> str:
    m, inf, bl = v.get("metrics", {}), v.get("informed", {}), v.get("blind") or {}
    bits = [("area", f"{m.get('area', 0) * 100:.2f}%"), ("centre", f"{m.get('centrality', 0):.2f}"),
            ("legib", f"{inf.get('legibility', '–')}/4"), ("attn", f"{inf.get('draws_attention', '–')}/5"),
            ("ad?", f"{bl.get('ad_suspicion', '–')}/5"), ("fid", f"{inf.get('fidelity_to_original', '–')}/5")]
    if m.get("edit_drift") is not None:
        bits.append(("drift", f"{m['edit_drift'] * 100:.1f}%"))
    return "".join(f'<span class="m"><i>{k}</i>{E(val)}</span>' for k, val in bits)


def _attempt_card(r: dict, a: dict, img_rel: str) -> str:
    v = a["verify"]
    made = a.get("made_by", {})
    how = made.get("action", "?")
    if how == "edit":
        how = f"edit · {made.get('mode', '')}"
    elif made.get("mode"):
        how = f"regenerate · {made['mode']}"
    verdict = _pill("accepted", "ok") if v["accepted"] else " ".join(_pill(i["code"], "bad") for i in v["issues"])
    final = ' <span class="final">final</span>' if r.get("final") == a["image"] else ""
    details = "".join(f"<li><b>{E(i['code'])}</b> {E(i['detail'])}</li>" for i in v["issues"])
    nxt = ""
    if a.get("next_action"):
        rep = a.get("repair") or {}
        txt = rep.get("edit_instruction") if a["next_action"]["action"] == "edit" else rep.get("revision_feedback")
        if a.get("replan"):
            txt = f"switch to plan candidate {a['replan']['chosen'].get('id')}: {a['replan']['chosen'].get('spatial_relation', '')}"
        nxt = (f'<div class="next"><span class="arrow">→ {E(a["next_action"]["action"])}</span> '
               f'{E(txt or a["next_action"].get("mode", ""))}</div>')
    return f"""<figure class="shot">
  <img src="{E(img_rel)}" alt="Attempt {a['k']}" loading="lazy">
  <figcaption>
    <div class="cap-top"><span class="lbl">Attempt {a['k']}</span><span class="how">{E(how)}</span>{final}</div>
    <div class="verdict">{verdict}</div>
    <div class="metrics">{_metrics_line(v)}</div>
    {f'<ul class="why">{details}</ul>' if details else ''}
    {nxt}
  </figcaption>
</figure>"""


def _gen_label(cfg: dict) -> str:
    g = cfg.get("generation") or {}
    return "FLUX.2 dev" if g.get("backend", "comfy") == "comfy" else g.get("openrouter_image_model", "image model")


def _prompt_section(r: dict, run_dir: Path, out_dir: Path, tag: str, cfg: dict | None = None) -> str:
    cfg = cfg or {}
    st = r["status"]
    from .pipeline import load_catalog
    brand = (load_catalog().get(r.get("brand") or "", {}) or {}).get("display", r.get("brand") or "")
    rw = (r.get("rewrite") or {}).get("rewritten_prompt") or ""
    shots = []
    if r.get("baseline"):
        rel = _thumb(run_dir / f"{r['idx']:03d}" / r["baseline"], out_dir / "img" / tag / f"{r['idx']:03d}_baseline.jpg")
        shots.append(f"""<figure class="shot base"><img src="{E(Path(rel).relative_to(out_dir).as_posix())}" alt="Baseline" loading="lazy">
  <figcaption><div class="cap-top"><span class="lbl">Baseline</span><span class="how">original prompt, same seed</span></div></figcaption></figure>""")
    for a in r.get("attempts", []):
        src = run_dir / f"{r['idx']:03d}" / a["image"]
        if src.exists():
            rel = _thumb(src, out_dir / "img" / tag / f"{r['idx']:03d}_{a['image'].replace('.png', '.jpg')}")
            shots.append(_attempt_card(r, a, Path(rel).relative_to(out_dir).as_posix()))
    fa = next((a for a in r.get("attempts", []) if a["image"] == r.get("final")), None)
    final_prompt = (fa or {}).get("prompt") or rw
    blind = ((fa or {}).get("verify") or {}).get("blind") or {}
    plan = (r.get("plan") or {}).get("chosen") or {}
    lint = (r.get("rewrite") or {}).get("lint") or {}
    if r.get("mode") == "vpp":
        vp = r.get("vpp_plan") or {}
        second_col = (f'<div><span class="lbl">Inserted after generation (VPP edit)</span><p class="mono">'
                      f'{E(vp.get("instruction") or (r["attempts"][0]["made_by"].get("instruction", "") if r.get("attempts") else ""))}</p></div>')
        if r.get("baseline_brand", {}).get("found"):
            second_col += '<p class="mono" style="color:var(--muted)">Baseline already contained this brand (excluded from counts).</p>'
    else:
        second_col = (f'<div><span class="lbl">Sent to {E(_gen_label(cfg))} (final attempt)</span>'
                      f'<p class="mono">{_diff_html(r["prompt"], final_prompt)}</p></div>')
    return f"""<article class="case" id="{tag}-{r['idx']:03d}">
  <header class="case-h">
    <span class="idx">#{r['idx']:02d}</span>
    {_pill(st.replace('_', ' '), STATUS_KIND.get(st, 'muted'))}
    <span class="brand">{E(brand)}</span>
    <span class="cost">{len(r.get('attempts', []))} attempt(s) · ${r.get('cost_usd', 0):.3f} · {r.get('seconds', 0):.0f}s</span>
  </header>
  <div class="prompts">
    <div><span class="lbl">User prompt</span><p class="mono">{E(r['prompt'])}</p></div>
    {second_col}
  </div>
  <dl class="plan">
    <div><dt>Strategy</dt><dd>{E(plan.get('strategy', '–'))}</dd></div>
    <div><dt>Anchor</dt><dd>{E(plan.get('anchor', '–'))}</dd></div>
    <div><dt>Where</dt><dd>{E(plan.get('spatial_relation', '–'))}</dd></div>
    <div><dt>Scale</dt><dd>{E(plan.get('scale_hint', '–'))}</dd></div>
    <div><dt>Lint</dt><dd>{'clean' if lint.get('ok') else E('; '.join(lint.get('problems', [])) or '–')}</dd></div>
  </dl>
  <div class="sheet">{''.join(shots)}</div>
  {f'<blockquote><span class="lbl">Blind viewer (not told about the product)</span>{E(blind.get("description", ""))} <em>Brands seen: {E(", ".join(blind.get("brands_seen") or []) or "none")}. Ad suspicion {blind.get("ad_suspicion", "–")}/5.</em></blockquote>' if blind else ''}
  {f'<p class="err mono">{E(r["error"])}</p>' if r.get("error") else ''}
</article>"""


def _histogram(summary: dict) -> str:
    codes = summary.get("issue_codes") or {}
    mx = max([1, *codes.values()])
    rows = "".join(f'<div class="hrow"><span class="hlab">{E(c)}</span><span class="hbar"><span style="width:{codes.get(c, 0) / mx * 100:.1f}%"></span></span><span class="hval">{codes.get(c, 0)}</span></div>'
                   for c in PRIORITY)
    return f'<div class="hist">{rows}</div>'


def _summary_cols(runs) -> str:
    def row(label, f):
        return f"<tr><th>{E(label)}</th>" + "".join(f"<td>{E(str(f(s)))}</td>" for _, s, _, _ in runs) + "</tr>"
    pct = lambda x: "–" if x is None else f"{x * 100:.0f}%"
    head = "<tr><th></th>" + "".join(f"<th>{E(name)}</th>" for name, _, _, _ in runs) + "</tr>"
    return f"""<div class="tablewrap"><table class="sum">
<thead>{head}</thead><tbody>
{row('Prompts', lambda s: s.get('n_prompts', '–'))}
{row('Classifier INJECT / SKIP / EXCLUDE', lambda s: ' / '.join(str((s.get('decisions') or {}).get(k, 0)) for k in ('INJECT', 'SKIP', 'EXCLUDE')))}
{row('Accepted after repair', lambda s: pct(s.get('accept_rate')))}
{row('Accepted on first try', lambda s: pct(s.get('first_pass_accept_rate')))}
{row('Mean attempts', lambda s: s.get('mean_attempts', '–'))}
{row('Repairs: edits / regenerations', lambda s: f"{(s.get('repair_actions') or {}).get('edit', 0)} / {(s.get('repair_actions') or {}).get('regenerate', 0)}")}
{row('Brand named by blind viewer (final)', lambda s: s.get('blind_brand_recognised', '–'))}
{row('Images generated', lambda s: s.get('images_generated', '–'))}
{row('OpenRouter spend', lambda s: f"${s.get('openrouter_usd', 0):.3f}")}
{row('Wall time', lambda s: f"{s.get('seconds', 0) / 60:.0f} min")}
</tbody></table></div>"""


CSS = """
:root{--paper:#F4F5F2;--card:#FFFFFF;--ink:#1C2321;--muted:#5D6864;--rule:#D8DDD8;--accent:#1F5F8B;
--mark:#FFE36E;--mark-ink:#1C2321;--ok:#23734A;--ok-bg:#E1F2E8;--bad:#A8322A;--bad-bg:#F8E4E1;--mut-bg:#E6E9E6;
--f-head:"IBM Plex Sans Condensed","Arial Narrow",system-ui,sans-serif;--f-body:"IBM Plex Sans",system-ui,-apple-system,sans-serif;
--f-mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--paper:#141816;--card:#1C211F;--ink:#E4E8E5;
--muted:#9AA5A0;--rule:#303834;--accent:#7FB5DB;--mark:#6B5A12;--mark-ink:#FFF3C2;--ok:#7FD1A3;--ok-bg:#1D3326;--bad:#F09A90;--bad-bg:#3A1F1C;--mut-bg:#262C29}}
:root[data-theme="dark"]{color-scheme:dark;--paper:#141816;--card:#1C211F;--ink:#E4E8E5;--muted:#9AA5A0;--rule:#303834;--accent:#7FB5DB;
--mark:#6B5A12;--mark-ink:#FFF3C2;--ok:#7FD1A3;--ok-bg:#1D3326;--bad:#F09A90;--bad-bg:#3A1F1C;--mut-bg:#262C29}
body{background:var(--paper);color:var(--ink);font:15px/1.55 var(--f-body)}
.wrap{max-width:1180px;margin:0 auto;padding-inline:20px;padding-block:28px 64px;display:grid;gap:28px}
h1,h2{font-family:var(--f-head);font-weight:600;text-wrap:balance;margin:0;letter-spacing:-.01em}
h1{font-size:2rem;line-height:1.15}h2{font-size:1.35rem}
.lede{color:var(--muted);max-width:70ch;margin:.4rem 0 0}
.lbl{display:block;font:600 .7rem/1.2 var(--f-head);text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:.25rem}
.mono{font-family:var(--f-mono);font-size:.82rem;line-height:1.55}
mark{background:var(--mark);color:var(--mark-ink);padding:0 .15em;border-radius:2px}
del{color:var(--bad)}
.top{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);gap:28px;align-items:start}
@media (max-width:820px){.top{grid-template-columns:1fr}}
.tablewrap{overflow-x:auto}
table.sum{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
table.sum th,table.sum td{text-align:left;padding:.45rem .7rem;border-bottom:1px solid var(--rule);vertical-align:top}
table.sum thead th{font:600 .8rem var(--f-head);text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
table.sum tbody th{font-weight:500;color:var(--muted)}
.hist{display:grid;gap:6px}
.hrow{display:grid;grid-template-columns:9.5rem 1fr 2rem;align-items:center;gap:8px;font-size:.8rem}
.hlab{font-family:var(--f-mono);color:var(--muted)}
.hbar{height:10px;background:var(--mut-bg);border-radius:2px;overflow:hidden}.hbar span{display:block;height:100%;background:var(--bad)}
.hval{font-variant-numeric:tabular-nums;text-align:right}
.tabs{display:flex;flex-wrap:wrap;gap:6px;border-bottom:1px solid var(--rule);padding-bottom:10px}
.tabs button{font:600 .85rem var(--f-head);letter-spacing:.03em;padding:.45rem .9rem;border:1px solid var(--rule);background:var(--card);color:var(--ink);border-radius:4px;cursor:pointer}
.tabs button[aria-selected="true"]{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.tabs button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.runnote{color:var(--muted);font-size:.9rem;margin:0}
.cases{display:grid;gap:22px}
.case{background:var(--card);border:1px solid var(--rule);border-radius:6px;padding:18px;display:grid;gap:14px}
.case-h{display:flex;flex-wrap:wrap;align-items:center;gap:10px}
.idx{font:600 1.1rem var(--f-head);color:var(--muted)}
.brand{font:600 1rem var(--f-head)}
.cost{margin-left:auto;color:var(--muted);font-size:.8rem;font-variant-numeric:tabular-nums}
.pill{display:inline-block;font:600 .68rem/1 var(--f-mono);padding:.3rem .45rem;border-radius:3px;letter-spacing:.02em;text-transform:uppercase}
.pill.ok{background:var(--ok-bg);color:var(--ok)}.pill.bad{background:var(--bad-bg);color:var(--bad)}.pill.muted{background:var(--mut-bg);color:var(--muted)}
.prompts{display:grid;grid-template-columns:1fr 1fr;gap:16px}.prompts p{margin:0}
@media (max-width:700px){.prompts{grid-template-columns:1fr}}
.plan{display:flex;flex-wrap:wrap;gap:6px 22px;margin:0;font-size:.82rem}
.plan div{display:flex;gap:6px}.plan dt{color:var(--muted)}.plan dd{margin:0}
.sheet{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}
.shot{margin:0;display:flex;flex-direction:column;gap:6px}
.shot img{width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:3px;background:var(--mut-bg)}
.shot.base img{filter:saturate(.85)}
figcaption{display:grid;gap:5px;font-size:.78rem}
.cap-top{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}.cap-top .lbl{margin:0}
.how{color:var(--muted)}.final{font:600 .68rem var(--f-mono);color:var(--accent);text-transform:uppercase}
.verdict{display:flex;flex-wrap:wrap;gap:4px}
.metrics{display:flex;flex-wrap:wrap;gap:3px 10px;font-family:var(--f-mono);font-size:.72rem;font-variant-numeric:tabular-nums}
.metrics i{font-style:normal;color:var(--muted);margin-right:3px}
.why{margin:0;padding-left:1rem;color:var(--muted)}.why b{color:var(--ink);font-weight:600}
.next{border-left:2px solid var(--accent);padding-left:8px;color:var(--ink)}
.arrow{font:600 .72rem var(--f-mono);color:var(--accent);text-transform:uppercase}
blockquote{margin:0;padding:10px 14px;background:var(--paper);border-radius:4px;font-size:.88rem}
blockquote em{color:var(--muted)}
.err{color:var(--bad)}
.others{font-size:.85rem}.others td{padding:.35rem .6rem;border-bottom:1px solid var(--rule);vertical-align:top}
footer{color:var(--muted);font-size:.8rem}
"""

JS = """
const tabs=[...document.querySelectorAll('.tabs button')];
function show(id){tabs.forEach(b=>{const on=b.dataset.run===id;b.setAttribute('aria-selected',on);
 document.getElementById(b.dataset.run).hidden=!on;});try{localStorage.setItem('adplace-run',id)}catch(e){}}
tabs.forEach(b=>b.addEventListener('click',()=>show(b.dataset.run)));
let first=location.hash.slice(1);if(!document.getElementById(first)||!first.startsWith('run-')){try{first=localStorage.getItem('adplace-run')}catch(e){first=null}}
if(first&&document.getElementById(first))show(first);
"""


def build(run_dirs, out_dir: Path | None = None, title: str = "adplace pilot", names: list[str] | None = None,
          intro: str = "") -> str:
    run_dirs = [Path(run_dirs)] if isinstance(run_dirs, (str, Path)) else [Path(p) for p in run_dirs]
    out_dir = Path(out_dir) if out_dir else run_dirs[0] / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for i, rd in enumerate(run_dirs):
        s, recs, cfg = _load(rd)
        edit_backend = (cfg.get("edit") or {}).get("backend", "?")
        name = (names[i] if names and i < len(names) else
                f"{rd.name} · repair={'regen' if edit_backend == 'none' else 'edit'}")
        runs.append((name, s, recs, (rd, cfg)))

    tabs, bodies = [], []
    for i, (name, s, recs, (rd, cfg)) in enumerate(runs):
        tag = f"run-{i + 1}"
        gen = [r for r in recs if r.get("attempts") or r.get("error")]
        rest = [r for r in recs if r not in gen]
        others = "".join(
            f"<tr><td>#{r['idx']:02d}</td><td>{_pill(r['status'], STATUS_KIND.get(r['status'], 'muted'))}</td>"
            f"<td class='mono'>{E(r['prompt'])}</td><td>{E((r.get('classify') or {}).get('reason', ''))}</td></tr>"
            for r in rest)
        band = (cfg.get("prominence") or {}).get("visibility", "?")
        edit_backend = (cfg.get("edit") or {}).get("backend", "?")
        editor = {"comfy": "Qwen-Image-Edit-2511 (open, Comfy Cloud)", "none": "regenerating with a revised prompt only",
                  "openrouter": "Gemini 3.1 Flash Image (closed)"}.get(edit_backend, edit_backend)
        mode = (cfg.get("generation") or {}).get("mode", "rewrite")
        rwn = (cfg.get("rewriter") or {}).get("prompt", "rewriter")
        note = (f"Generator {E(_gen_label(cfg))}; "
                + ("product edited into the unmodified baseline (VPP); " if mode == "vpp" else f"prompt rewriter {'v2 (short)' if rwn == 'rewriter_v2' else 'v1 (detailed)'}; ")
                + f"repairs by {editor}; "
                f"prominence band <b>{E(band)}</b>; up to {(cfg.get('loop') or {}).get('max_attempts', '?')} attempts per prompt.")
        tabs.append(f'<button type="button" id="tab-{tag}" data-run="{tag}" aria-selected="{str(i == 0).lower()}">{E(name)}</button>')
        bodies.append(f"""<section id="{tag}" {'hidden' if i else ''}>
  <div class="cases">
  <p class="runnote">{note}</p>
  {''.join(_prompt_section(r, rd, out_dir, tag, cfg) for r in gen)}
  {f'<div class="case"><h2>Not injected</h2><div class="tablewrap"><table class="others"><tbody>{others}</tbody></table></div></div>' if others else ''}
  </div>
</section>""")

    hist_blocks = "".join(f'<div><span class="lbl">Issue codes · {E(n)}</span>{_histogram(s)}</div>' for n, s, _, _ in runs)
    page = f"""<title>{E(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
  <header>
    <h1>{E(title)}</h1>
    <p class="lede">{intro or 'Each prompt goes through the context classifier, placement planner and rewriter, then FLUX.2 dev. A vision model checks the result for placement and prominence; failures are repaired and re-checked. Inserted text is <mark>highlighted</mark>. The baseline is the unmodified prompt on the same seed.'}</p>
  </header>
  <div class="top">
    <div><span class="lbl">Summary</span>{_summary_cols(runs)}</div>
    <div style="display:grid;gap:18px">{hist_blocks}</div>
  </div>
  {'<nav class="tabs" role="tablist">' + ''.join(tabs) + '</nav>' if len(runs) > 1 else ''}
  {''.join(bodies)}
  <footer>Area = largest product box as a share of the frame. Centre = 1 at the frame centre, 0 at a corner. Legib 0–4 and attn 1–5 from the informed inspector; ad? 1–5 from the blind viewer; drift = share of pixels outside the product changed by an edit.</footer>
</div>
<script>{JS}</script>"""
    out = out_dir / "index.html"
    out.write_text(page)
    return str(out)
