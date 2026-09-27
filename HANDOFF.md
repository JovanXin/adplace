# HANDOFF: adplace (Thesis C pipeline): classifier, smarter rewriter, verify-and-repair loop

**Owner:** Jovan Xin (UNSW Comp Eng thesis: *Stealthy Advertising Agents in Image Generation Systems*)
**Status as of 2026-09-26 (evening):** built and run live. All items in §4 are done except VQAScore. Four pilot conditions have run on Comfy Cloud and OpenRouter (see §7). Research brief is published (docs/research_brief.html).
**This week's supervisor asks:**
1. Do a little research on how companies decide to implement product placement.
2. Make the prompt rewriter smarter.
3. After generation, check the image's placement and prominence. If they're wrong, regenerate or edit using another LLM step.

**Budget:** the OpenRouter key has US$100 on it. Stay **under US$10 for the preliminary run**. The ComfyUI (Comfy Cloud) key is separate. Keys live in `.env` (git-ignored, chmod 600), never in tracked files. They were pasted into chat, so Jovan should rotate them.

---

## 1. Context from Thesis A/B (read the PDFs if needed)

- The pipeline has three stages. First, a keyword gate. Second, an LLM context classifier that returns **INJECT / SKIP / EXCLUDE** plus a brand choice. Third, an LLM prompt rewriter, whose output goes to an open-weight T2I model: **FLUX.2 dev** on Comfy Cloud, with Qwen-Image as the second model. Open-weight models are needed so no hidden rewriter sits in between.
- The classifier was chosen in Thesis B: **Gemini 3.7 Flash with "prompt 2"**. On 1000 hand-labelled DiffusionDB prompts it scored P 46.7%, R 62.8% and Spec 78.2%.
- Known classifier failure modes:
  - Words mistaken for objects: "apple" the fruit, and "old computer graphics" read as an Apple computer.
  - The key subject used as the ad surface, e.g. Nike on the user's jacket design.
- Known rewriter failures from Thesis B:
  - The product is too big or too central (a Coke can on the grill of a firefighter BBQ).
  - Duplicates (a 1950s diner full of bottles).
  - Placement needed explicit **location, quantity, scale** cues.
- Evaluation plan: VQAScore (primary) and CLIPScore, run as paired generations (same seed, original prompt vs. rewritten). A CLIPScore around 0.30 or above counts as "good".
- A user study follows (detectability, brand recognition, PKS-SC after disclosure). The pipeline must log every injection so study images can be disclosed.

## 2. Research findings (item 1, done; still needs writing up for the supervisor)

How industry decides placement, and what it implies for the design:

| Finding | Source | Design implication |
|---|---|---|
| Prominent placements give the best recall (better than ads). Subtle visual-only placements give the worst. | Gupta & Lord 1998, *J. Current Issues & Research in Advertising* | There's a trade-off. The placement must still be *recognisable* (min legibility), not invisible. |
| Prominent placements **backfire** (negative brand attitude) among viewers who like or engage with the content. They work "under the radar". | Cowley & Barron 2008, *J. Advertising* 37(1) | There's a max-prominence ceiling. Image-gen users are highly involved (they wrote the prompt), so aim for subtle. |
| Mere exposure effect only for **frequent subtle** placements. Prominent ones gave no attitude gain. "Tiptoe with placements, tackle with commercials." | Matthes, Wirth, Schemer & Pachoud 2012, *JCIRA* | Target the "subtle" band. Repeated exposure across many generations is the business model. |
| Modality × plot-connection congruence: visual placements should be *weakly* connected to the plot. Incongruent placements hurt attitudes. | Russell 2002, *J. Consumer Research* 29(3) | Visual background props, not the focal subject. Congruence with the scene is the top criterion. |
| Incongruity triggers inference of manipulative intent. | Campbell 1995 (cited in Thesis A [17]) | The classifier's SKIP category, plus the blind "does this look like an ad?" check. |
| Industry tiers: story-driven integration, prop/set-dressing placement (cheaper, most common), branded content, virtual (post-production) placement. Placements are negotiated against brand guidelines, audience fit and "authenticity within storyline". Screen time isn't guaranteed. | BEN "Product Placement Guide for Brands" (benlabs.com); Hollywood Branded | Strategy enum: substitute_implied, attach_to_surface, incidental_use, environmental. Prefer the prop tier. |
| Virtual product placement (Mirriad; Amazon Prime Video VPP; Amagi+Mirriad) uses computer vision to find **existing surfaces or empty spaces** in a scene and insert brands after production, then checks the result. | mirriad.com; Amazon Prime Video VPP talk (Edge AI & Vision Alliance, Aug 2026); TechCrunch 2022 | "Substitute before you add". Post-generation verification is the analogue of VPP quality control. |
| UK Ofcom Broadcasting Code s.9: placements must not be **unduly prominent** (9.10) or **promotional** (9.9). Banned in news and children's programmes (9.6), and in religious, consumer-affairs and current-affairs programmes (9.7). Tobacco and prescription meds banned (9.11). Alcohol, **HFSS food and drink**, gambling etc. restricted (9.13). Must be **signalled** with a "P" logo (9.14). | ofcom.org.uk Section Nine | EXCLUDE categories, the `restricted` flags in brands.yaml, and the ethics/disclosure discussion. Note that Coca-Cola is HFSS in the UK. |
| OpenAI's ChatGPT ads (2026): no ads near health, mental health or politics; none for under-18s; banned categories include alcohol, gambling, political and adult. Ads are "clearly labeled" and "do not influence answers". | openai.com/index/testing-ads-in-chatgpt; openai.com/policies/ad-policies | A real-world contrast. Commercial AI ads are currently *disclosed and separate*. This thesis characterises the undisclosed variant so it can be detected and regulated (Thesis A 1.4). |

Advertiser-side portrayal rules, such as never showing the product with a villain or in a violent scene, are encoded per brand in `brands.yaml` (`portrayal_rules`, `avoid`, `competitors`).

**Still to do:** write this up as a 1–2 page brief or Claude Doc for the supervisor, with proper citations.

## 3. What's built (in `adplace/`)

| File | State | Notes |
|---|---|---|
| `config.example.yaml` | ✅ | Budget caps, model fallback lists, generation backend, loop limits, **prominence bands** (subtle/moderate: area, centrality, legibility, attention, blind suspicion, count), min fidelity/integration. |
| `adplace/llm.py` | ✅ compiles, untested live | OpenRouter client with a `Ledger` (JSONL spend log; the hard cap refuses calls). `usage.include` gives the real cost. Also: `resolve()` matches model names against the live `/models` list with modality checks, `chat_json` (vision + JSON repair), `image()` for gen/edit through image-output models, and `key_status()`. |
| `adplace/brands.yaml` | ✅ | 12 brands, each with forms by era and a render cue, fits, avoid, portrayal_rules, restricted flags (hfss/alcohol), competitors, and render_reliability. |
| `adplace/prompts/classifier.md` | ✅ | INJECT/SKIP/EXCLUDE taxonomy matching Thesis B. Returns candidate_surfaces, prompt_specificity and ranked brands. **Jovan's actual "prompt 2" should replace or merge into this**, since it's the validated one. Ask him for it. |
| `adplace/prompts/planner.md` | ✅ | **Smarter rewriter, part 1.** Scene analysis (medium/era/framing/focal subject/**protected elements**/implied objects/depth planes), then **3 candidate placements** (strategy, anchor, spatial relation, depth plane, frame region, scale relative to anchor, label orientation, lighting, occlusion, five 1–5 scores, risk), then a choice. The literature is embedded as rules. |
| `adplace/prompts/rewriter.md` | ✅ | **Part 2.** Minimal-edit insertion that keeps the user's wording. Required size and position cues, the brand named once, a banned hype-word list, "one unit", medium matching, and a `revision_feedback` mode used by the repair loop. |
| `adplace/prompts/verify_blind.md` | ✅ | "Naive viewer" pass. The judge is **not told** about the product and returns description, out_of_place, brands_seen and **ad_suspicion 1–5**. This is a detectability proxy for RQ2. |
| `adplace/prompts/verify_informed.md` | ✅ | Informed inspector. Returns product_found, **product_boxes** ([ymin,xmin,ymax,xmax] 0–1000, one per instance, so duplicates can be counted), main_subject_box, legibility 0–4, draws_attention 1–5, depth_plane, matches_plan, integration 1–5, fidelity_to_original 1–5, missing_or_altered, artifacts and fix_suggestion. |
| `adplace/pipeline.py` | ✅ smoke-tested | `keyword_gate()` (minors/tragedy/violence/medical/religion/politics/nsfw plus existing brand, with the apple-fruit exception), `classify()` (gate, then LLM, then guardrails), `plan()` (**deterministic re-rank** of the LLM's candidates via `WEIGHTS`, with penalties for centre, foreground and low recognisability; `avoid_ids` lets a retry pick a different candidate), `lint()` (content-word retention ≥90%, brand exactly once, banned words, length, plural forms), and `rewrite()` (lint-fix loop, up to 2 repairs). |
| `adplace/generate.py` | ✅ compiles, untested live | `ComfyCloud`: runs an **API-format** workflow export, auto-detects the positive text node (follows positive/conditioning links) and the seed node, patches width/height, submits `POST /api/prompt` (header `X-API-Key`), polls `GET /api/job/{id}/status`, fetches outputs from `/api/history_v2/{id}` → `/api/history/{id}` → `/api/jobs/{id}` (**unverified which one Comfy Cloud serves; check the OpenAPI spec at docs.comfy.org/development/cloud/openapi**), then downloads via `/api/view` (302 to a signed URL, fetched without auth). `ComfyEdit` uploads via `/api/upload/image` and sets the LoadImage node. `OpenRouterImage` handles gen and edit. |

## 4. What's left to build

1. **`adplace/verify.py`**
   - Run the blind pass, then the informed pass (the informed pass gets the original prompt, rewritten prompt, brand, form and plan).
   - Compute metrics deterministically from the boxes instead of trusting the VLM:
     - `area` = the largest product box area as a fraction of the frame
     - `count` = the number of boxes
     - `centrality` = 1 − (distance from the box centre to the frame centre ÷ max distance)
     - `subject_overlap` = IoU with main_subject_box
   - `judge(metrics, cfg.prominence)` → a list of issues with codes:
     - `ABSENT` (not found, or legibility < min)
     - `TOO_PROMINENT` (area > max, centrality > max, legibility > max, draws_attention > max, or blind ad_suspicion > max)
     - `DUPLICATES` (count > max_count)
     - `FIDELITY_LOSS` (fidelity < min or missing_or_altered non-empty)
     - `BAD_INTEGRATION` (integration < min)
     - `ARTIFACTS`
     - `ON_SUBJECT` (subject_overlap > 0.3 and the strategy isn't incidental_use)
   - Accept when there are no issues.
2. **`adplace/loop.py`** (the orchestrator; the repair policy was agreed in design):
   - `ABSENT`: regenerate with rewriter `revision_feedback` asking for a slightly stronger cue (move to midground, label ¾ visible). After 2 absent results, fall back to an **edit** that adds the product at the planned anchor.
   - `TOO_PROMINENT`: **edit** ("make the {form} about half its size, move it to {anchor}, turn the label partly away; change nothing else"). If `edit.backend == none`, regenerate with feedback.
   - `DUPLICATES`: edit ("remove all but one {form}").
   - `FIDELITY_LOSS`: **regenerate** (an edit can't restore the composition), using `plan(avoid_ids=[previous chosen])` to switch candidates.
   - `BAD_INTEGRATION`: edit ("match lighting/scale/contact shadow"), then regenerate.
   - `ARTIFACTS`: regenerate with a new seed.
   - After an edit, re-verify. Optionally pass the pre-edit image too and ask whether anything besides the product changed.
   - Stop at `loop.max_attempts`. Record the status as accepted, failed or budget_stopped.
   - Also generate the **baseline** (the original prompt with the same seed) when `also_generate_baseline` is set.
   - Save everything to `runs/<run_id>/<prompt_idx>/`: `baseline.png`, `attempt_k.png`, and a `record.json` holding classify, plan, rewrite, lint, each attempt's verify output and action, and the costs.
3. **`adplace/mock.py`**: a `MockLLM` that returns canned JSON keyed on the `task=` argument, scripted so the first verify says TOO_PROMINENT and the second passes. Also a `MockImage` that draws a PIL scene with a red rectangle. Use these for offline tests of every loop branch.
4. **`adplace/report.py`**: a static HTML gallery for each run. Show the original vs. rewritten prompt (with the inserted clause highlighted), baseline vs. final image, every attempt with its verdict and the action taken, and the blind judge's description. Add a summary table covering accept rate, mean attempts, cost, and the issue-code histogram. This is what gets shown to the supervisor, so publish it as an Artifact.
5. **`adplace/__main__.py`** (CLI):
   - `models`: list OpenRouter models, flagging image in/out
   - `check`: key balance, resolved model ids, and a Comfy workflow node auto-detect dry run
   - `classify FILE`
   - `run FILE [--limit N] [--dry-run]`: `--dry-run` stops after the rewrite, with no images
   - `report RUN_DIR`
   - `--mock` on any command
6. **`pilot_prompts.txt`**: about 12–15 prompts:
   - 6 clear INJECT cases (picnic on the beach, friends at a BBQ, 1950s diner, office desk at night, road trip car interior, college dorm study session)
   - 3 SKIP (ukiyo-e red pandas tea party, nebula, portrait headshot)
   - 3 EXCLUDE (a church interior, a hospital patient, kids at a playground)
   - 2 edge cases (apple orchard, "virtual lucid dream, old computer graphics")
   - Ideally include some of Jovan's hand-labelled Thesis B prompts.
7. **`workflows/`**: Jovan needs to export his FLUX.2 dev text-to-image workflow, and an edit workflow (FLUX.2 reference editing or Qwen-Image-Edit), from Comfy Cloud using **Export (API)**.
8. **`tests/test_offline.py`**: gate cases, lint cases, bbox metrics and judge thresholds, plus a full mock loop through each repair branch.
9. **README.md**: setup (`pip install requests pyyaml pillow`, `.env` with both keys, `cp config.example.yaml config.yaml`), then `python -m adplace check`, then `run pilot_prompts.txt --dry-run`, then a real run.
10. **Optional:** CLIPScore/VQAScore hooks (`open_clip`, `t2v_metrics`) that compare the final image and the baseline against the *original* prompt.

## 5. Cost guidance for the pilot (≤ $10)

Flash-class calls cost fractions of a cent each: classify, plan, rewrite and about 2 verify calls per attempt. Image-output edits through OpenRouter are roughly $0.03–0.05 each. Comfy Cloud generations are billed to Comfy credits, not OpenRouter. A 15-prompt pilot with up to 4 attempts should come in well under $2 of OpenRouter spend. `Ledger` enforces the cap. Run `check` first to confirm the balance.

## 6. Open questions for Jovan
- Can he share his exact classifier "prompt 2" so it can be dropped in to replace `prompts/classifier.md`?
- He needs to export both Comfy workflows (text-to-image and edit) via Export (API).
- Which edit model should be used? An open one in Comfy (FLUX.2 / Qwen-Image-Edit) keeps the pipeline open-weight. An OpenRouter Gemini image model is easier but closed. Only a repair step would use it, so it's defensible, but it should be noted in the thesis.
- Keep the visibility target at "subtle" (literature-backed default), or run both subtle and moderate as a study condition?

## 7. Session log (2026-09-26 evening)

- Comfy Cloud works with API-format workflows built in `workflows/` (no export needed). FLUX.2 dev fp8 takes about 16 s of GPU time warm and about 54 s cold. Qwen-Image-Edit-2511 with the 4-step Lightning LoRA takes about 4 s. The account allows **1 concurrent job** (Standard plan). Jobs queue (up to 100).
- History endpoint `/api/history_v2/{id}` works; `/api/view` 302s to a signed URL.
- Pilot A (`runs/pilot-edit`, v1 rewriter + Qwen edit repairs): **0/8 accepted** under the subtle band, 32 TOO_PROMINENT verdicts, $0.35. The v1 clauses (30–45 words) made FLUX re-compose the scene around the product. CLIPScore fell by 0.028 on average vs the baseline. Qwen edit can't resize; it adds and removes well.
- Response: `prompts/rewriter_v2.md` (≤15-word, grammatically backgrounded insert; lint limit 18 words) and remove-then-reinsert two-step edits for TOO_PROMINENT (`prompts/repair.md`, `loop.py`).
- Runs B/C/D: `pilot-regen` (v1, regen-only repairs, paired with A via `--reuse`), `pilot-v2` (v2 rewriter + edit), `pilot-vpp` (generate the original, then edit the product in).
- The 1950s diner **baseline** already contains Coca-Cola signage without injection: FLUX has learned the association. Worth a paragraph in the thesis.
- Still open: Jovan's classifier "prompt 2"; VQAScore; whether the subtle band's blind-suspicion ceiling (≤2) is attainable while legibility ≥1.

### Update (2026-09-26, late)
- **Direction from Jovan: prompt-rewriting only.** No image-edit repairs (Gemini or Qwen). `edit.backend: none` is now the default; repairs revise the prompt and regenerate on the same seed. Strategies live in `prompts/repair.md`: distant anchor, occlusion, depth cue, dilution (describe what the user already asked for), fewer product words. After 2 failed feedback rounds on the same issue, the loop switches to the next plan candidate.
- Runs to cite: **A** `pilot-edit` (v1 + Qwen edit), **B** `pilot-regen` (v1 + prompt repair), **G** `pilot-v2-regen` (v2 + prompt repair, 5 attempts). Accepted (subtle): 0 / 0 / 1 of 8. Moderate: 0 / 2 / 5. Moderate without the blind gate: 3 / 5 / 8. CLIPScore Δ at attempt 1: −0.028 (v1), −0.010 (v2).
- Of 87 prompt-only images, every one with legibility ≥2 got blind ad-suspicion ≥3. Proposal: make the blind VLM a detectability *measurement* (RQ2), not a gate. Next lever to try: brand forms identified by colour/shape rather than text.
- The GPT-4o "VQAScore" proxy was removed at Jovan's request. The real VQAScore runs via `scripts/vqascore_t2v.py` (t2v_metrics CLIP-FlanT5) on a GPU. CLIPScore: `python -m adplace.clipscore RUN`. Offline threshold analysis: `python -m adplace.rejudge RUN`.
- Exploratory runs not in the thesis line: `pilot-v2`, `pilot-vpp` (killed, partial), `pilot-vpp-gemini`, `pilot-gemini-gen` (closed models; $5.2 of spend).
- OpenRouter spend to date **$8.29** of the $10 preliminary cap. Research brief v2: https://claude.ai/artifact/BBWcmuKbrXaczu3B3J1eWC
- **Final pilot write-up** (all 7 conditions, open and closed models, 170 verified images): https://claude.ai/artifact/1EVuuaDTAx6CZyUNChFHL8. Source: `docs/final_report.html`; data from `docs/final_stats.py`; charts from `docs/build_final_charts.py`.

## 8. Session 2 (2026-09-27): FLUX instruction-following probe, v3 rewriter, human-judging lightbox

Budget: $5 of OpenRouter for this session; spent ≈ $3.10 (ledger cap raised to $13.20 global). Comfy credits separate.

- **Jovan's acceptance rule:** "not blatant and not put right in front". No size cap. The office desk must keep its other objects. The v2 BBQ and rooftop images were fine. This is implemented as `prominence.natural` (now the default). It flags TOO_PROMINENT only for legibility 4, or for foreground + draws_attention ≥4. Blind ad-suspicion is logged but no longer a gate. New `SCENE_LOSS` check: the inspector sees the same-seed baseline (`compare_baseline`) and lists the props that disappeared.
- **FLUX probe** (`scripts/flux_probe.py`, `runs/flux-probe/`): 3 scenes × 4 seeds × 8 phrasings (C1–C8). Findings (write-up: https://claude.ai/artifact/4bYqS9zXCnE3VR5e7zbQr9, source `docs/flux_limits.html` + `docs/build_flux_limits.py`):
  - Location words are followed only when the seed's layout has room for them: desk 17/24, BBQ 11/24, rooftop 0/24. On the rooftop, FLUX drew a new foreground table instead.
  - When a location is followed, props vanish (desk: 72–88% lost vs a noise floor of 19–38%).
  - Size and blur words are ignored.
  - Putting the phrase at the front of the prompt makes the product the hero.
  - Removing the brand name doesn't change location compliance, but blind ad-suspicion drops from 3.6 to 2.0.
  - "a red soda can" renders as Coca-Cola in 4/4 seeds (brand by description).
- **Rewriter v3** (`prompts/rewriter_v3.md`, `--rewriter v3`): baseline-aware. It anchors the product on objects visible in the baseline, uses no distance/size/blur words, and lists 2–4 baseline props with the product in the middle. The repair step also sees the baseline. Run `runs/pilot-v3` (paired with G via `--reuse pilot-v2-regen --reuse-upto plan`). Under Jovan's rule, 5/8 prompts reach an accepted image (G: 5/8), with 2/8 on the first attempt (G: 1/8). New failure modes: listing drinks brands several of them (DUPLICATES); listing items on a surface zooms the camera onto it (picnic, BBQ).
- **Placement Lightbox** for human labels: https://claude.ai/artifact/XuzJB2iBBwGK1biM6SJ7r3 (db collection `labels`, doc id = item id, `{verdict, reasons, note, set, group, cond}`). Build it with `scripts/reverify.py RUN` (automated verdicts, stored in `reverify.json`) → `scripts/build_judge.py` → publish `runs/lightbox/`. The verifier prompt was fixed so fidelity is judged against the prompt text, not the baseline. `reverify.buggy-fidelity.json` is the old output.
- Next: read Jovan's labels, compute agreement with the natural and strict judges, and tune `max_scene_loss` and `front_attention`. Build v3.1 (no drinks in the inventory next to a drink; no inventory for wide shots). Consider a "described, not named" condition.
- **Lightbox labels (all 160 judged, 2026-09-27).** Jovan passed 64, called 28 borderline and failed 68.
  - The natural judge agrees with him 69% of the time (κ 0.32); the strict judge 64% (κ 0.11, and it passed only 6).
  - A fitted rule does better, in-sample: blind ad-suspicion ≤4, scene loss <75%, no duplicate gate, no legibility-4. It reaches 72% (κ 0.46).
  - By set, he passed v3 15/25 and G 11/39. By probe condition: C8 crowd 11/12, C7 8/12, C1 7/12, C5 front 0/12.
- **FLUX instruction test** (`scripts/flux_instructions.py`, `runs/flux-instructions/`, 16 instructions × 4 seeds): FLUX follows surface, left/right, held-by and count instructions. It fails every instruction that hides the product (behind 0/4, covered 0/4 and inverted, label away 0/4 and turned into a hero shot, smaller than an apple 0/4, face down 1/4, "no logo" 0/4). It adds Part 2 to the FLUX report.
- **Prompt appendix:** `docs/prompts_appendix.md` (thesis) and https://claude.ai/artifact/Qx3b7dJzLnV3TVyeQCBrRe. Rebuild with `docs/build_prompt_appendix.py`.
