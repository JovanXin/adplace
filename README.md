# adplace

Research prototype for *Stealthy Advertising Agents in Image Generation Systems* (UNSW thesis, Jovan Xin).
A prompt-layer pipeline that decides whether a brand fits an image prompt, plans a subtle placement,
rewrites the prompt, generates with an open-weight model, **checks the result for placement and
prominence, and repairs it** when it is missing, too prominent, duplicated, on the subject, or broke the scene.

```
prompt ─► keyword gate ─► classifier (INJECT/SKIP/EXCLUDE + brand)
        ─► planner (scene analysis → 3 candidates → deterministic re-rank)
        ─► rewriter (+ lint: retention, brand once, no negation, clause position, no plurals)
        ─► FLUX.2 dev on Comfy Cloud   (+ baseline: original prompt, same seed)
        ─► verify: blind viewer (ad suspicion) + informed inspector (boxes, legibility, attention, fidelity)
                  → metrics from boxes (area, centrality, count, subject overlap, edit drift)
                  → judge against the prominence band → issue codes
        ─► repair: vision LLM writes an edit instruction (Qwen-Image-Edit-2511) or rewriter feedback (regenerate)
        ─► re-verify … up to loop.max_attempts
```

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install requests pyyaml pillow pytest
```

Create `.env` next to this file (it is git-ignored; never commit keys):

```
OPENROUTER_API_KEY=sk-or-...
COMFY_API_KEY=comfyui-...
```

Optionally `cp config.example.yaml config.yaml` and edit. Without `config.yaml` the example config is used.

## Commands

```bash
.venv/bin/python -m adplace check                          # key balance, model ids, Comfy node auto-detect
.venv/bin/python -m adplace classify pilot_prompts.txt     # classifier only
.venv/bin/python -m adplace run pilot_prompts.txt --dry-run --name dry1      # stop after rewrite (≈$0.16 for 17 prompts)
.venv/bin/python -m adplace run pilot_prompts.txt --name pilot-edit --repair edit --reuse dry1
.venv/bin/python -m adplace run pilot_prompts.txt --name pilot-regen --repair regen --reuse pilot-edit
.venv/bin/python -m adplace report runs/pilot-edit
.venv/bin/python -m pytest -q tests                         # offline, uses MockLLM/MockImage
```

Options on `run`:

| flag | meaning |
|---|---|
| `--repair edit` | repairs use Qwen-Image-Edit-2511 on Comfy Cloud (open weights); FIDELITY_LOSS/ARTIFACTS still regenerate |
| `--repair regen` | never edit: every repair rewrites the prompt with the verifier's feedback and regenerates on the same seed |
| `--mode vpp` | "virtual product placement": generate the original prompt, then edit the product in (the image model never sees the brand) |
| `--visibility moderate` | use the moderate prominence band instead of subtle |
| `--reuse RUN` | copy classify/plan/rewrite/baseline/attempt 1 from an earlier run, so two repair conditions are paired |
| `--mock` | (global flag, before `run`) fully offline |

Outputs go to `runs/<name>/`: one folder per prompt (`baseline.png`, `attempt_k.png`, `record.json`
with every stage's output, each verify result and each repair action), plus `summary.json`,
`injections.csv` (for post-study disclosure) and `report/index.html`.

## Budget

`runs/ledger.jsonl` logs every OpenRouter call with its real cost (`usage.include`). The cap
`budget.openrouter_usd_cap` is **global across runs**, and calls are refused once it is reached.
`budget.max_image_generations` caps Comfy jobs per run. Rough costs with Gemini 3.7 Flash: classify ≈ $0.004,
plan ≈ $0.009, rewrite ≈ $0.004, verify (blind + informed) ≈ $0.006, repair ≈ $0.003. Comfy Cloud
jobs are billed to Comfy credits and run one at a time on this account (FLUX.2 dev ≈ 55 s, Qwen edit ≈ 80 s).

## Files

| path | what |
|---|---|
| `adplace/pipeline.py` | keyword gate, classifier, planner + re-rank, rewriter + lint |
| `adplace/verify.py` | blind/informed VLM passes, box metrics, edit drift, judge |
| `adplace/loop.py` | orchestrator, repair policy, repair writer, run summary, disclosure log |
| `adplace/generate.py` | Comfy Cloud client (API-format workflows), OpenRouter image backend |
| `adplace/report.py` | static HTML report (one or several runs) |
| `adplace/mock.py` | offline MockLLM / MockImage |
| `adplace/prompts/*.md` | classifier, planner, rewriter, verify_blind, verify_informed, repair |
| `adplace/brands.yaml` | advertiser catalogue (forms by era, fits, avoid, portrayal rules, restricted flags) |
| `workflows/` | `flux2_dev_api.json` (text-to-image), `qwen_edit_2511_api.json` (edit) |
| `docs/research_brief.html` | how companies decide on product placement, and how it maps onto the pipeline |
