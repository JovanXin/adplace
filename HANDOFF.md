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

## 9. Session 3 (2026-10-04): survey image set on closed models

Budget $10 (OpenRouter); spent **$8.88** (ledger cap is now $21.60 global). **Comfy Cloud returns 402 "Subscription required"**, so FLUX can't be used until it's renewed. Jovan chose closed image models.

- **Survey design (Jovan):** participants see the non-rewritten and rewritten images and answer about both; nobody is told which one has the ad. Deliverable: 16 items × {no-inject, bad-inject, good-inject} in `survey/images/` (`S01_A..S16_C`, letters shuffled), with the key in `survey/key.csv`. Researcher page: https://claude.ai/artifact/3c5jeod9ZV6JjrKSYxhgqX (source `docs/survey_set.html`, built by `docs/build_survey_page.py` from `docs/survey_set_template.html`).
- **Closed-model seed test** (`scripts/closed_seed_test.py`):
  - Gemini 3.1 Flash and Flash-Lite repeat exactly on a seed, but any prompt change re-draws the scene, so there's no paired design.
  - Gemini 2.5 and GPT-5 Image Mini aren't repeatable at all.
  - Images come back as JPEG. `image_to_data_url` now sniffs the MIME type, because Anthropic rejects a mislabelled type with a 400. Aspect ratio is pinned to 1:1.
- **Generator:** `google/gemini-3.1-flash-lite-image` ($0.034, ~4 s).
- **Rewriter v4** (`prompts/rewriter_v4.md`): for instruction-following models. The product goes mid-inventory, with at most one concealment cue and ≤18 words. Closed models follow these cues, whereas FLUX didn't, so v4 drops v3's ban on position words.
- **Benchmark** (`scripts/survey_bench.py`, `runs/survey-closed/`, 21 prompts; verify uses the `calibrated` band; judge is GPT-5.6 Sol, `prompts/survey_judge.md`):
  - Survey-good rate: naive 3/21, v2 4/12, v4-Gemini-Flash 6/21, v4-Claude-Sonnet-5.5 6/20.
  - Feedback loop (judge-aware repair → v4 rewrite) rescued 3 of the 8 prompts that had no usable image.
  - GPT ad-likelihood: naive 3.33, v4f 2.76, bad 4.19.
  - CLIPScore: base 0.314, v4f 0.306.
- **Pairwise survey proxy** (`scripts/survey_select.py pairwise`): shown side by side with the baseline, GPT picks the good-inject as the ad image in 13/16 pairs, and the bad-inject in 15/16. The side-by-side presentation drives detection, so the survey design should account for it.
- **New config/CLI:** `prominence.calibrated` band (the rule fitted to the lightbox labels); `--rewriter-model`, `--brand-prefs`, `--visibility calibrated`. `survey_prompts.txt` and `survey_brand_prefs.json` are in the repo root.

### Session 3b (2026-10-04): simple vs surface-specific rewriter, which replaces the good/bad set
- **Jovan's correction:** the conditions aren't good/bad inject. They are the rewriter strategies: **simple** ("…, with a can of Coke" / "with a Nike shirt") vs **surface-specific** ("…, with a can of Coke on the cafe table"), and surface-specific is the target. The good/bad set was moved to `survey_v1_good-bad/`, and `survey/` now holds the v2 set.
- `scripts/survey_surface.py` (`runs/survey-surface/`) reuses the 21 prompts plus 3 new ones (24) and the brand-free originals from `runs/survey-closed`. One LLM call (`prompts/placement_phrase.md`, sees the original image) writes `product` and `surface`, so the two rewrites differ only by the surface clause. Feedback loop: a failed surface image leads to a new surface chosen from the image and the issues (≤2 tries). Also tested `surface_short`, a short relational reference ("on their table").
- **Results (24 prompts, Gemini 3.1 Flash-Lite, calibrated rule):**
  - Pass rate: simple 16/24, surface 15/24, surface-short 11/24, surface + loop **20/24**.
  - GPT naturalness: simple 4.26, surface 3.65.
  - Side-by-side GPT spotted the rewrite: simple 18/24, surface 22/24.
  - CLIPScore: base 0.314, simple 0.312, surface 0.311.
  - Naming a surface makes the model **stage** it: it brings the towel or table into the foreground with the product on it. The surface only wins with the feedback loop. Cup holders and similar spots where the camera already looks work well.
- Export: 20 items × {no-inject, simple, surface} in `survey/images` + `survey/key.csv`. Two letterboxed simple images were re-rolled (seeds 8 and 9). Page (same URL): https://claude.ai/artifact/3c5jeod9ZV6JjrKSYxhgqX, built by `docs/build_survey_page2.py`.
- Spend: $4.73 of the $6 (ledger ≈ $25.25, cap $26.50).

## 10. Session 4 (2026-10-26): VQAScore on the study pairs (Modal)

- `scripts/vqascore_modal.py` runs canonical VQAScore (**CLIP-FlanT5-XXL**, `t2v_metrics==1.2`, the last release with CLIP-FlanT5; 3.0 dropped it) plus CLIPScore ViT-B/32 on a Modal L40S. Weights are cached in the Modal Volume `adplace-hf-cache`. Run it with `modal run scripts/vqascore_modal.py [--pairs study_pairs_all] [--limit N]`. Output: `<pairs>/vqascore_clip-flant5-xxl.csv` + `_summary.json`. The image pins are hand-resolved (open-clip 2.24 because ImageReward needs timm 0.6.13; hub 0.23.4; diffusers 0.27.2; datasets 2.19.2 / pyarrow 15).
- Cost: a full 171-image run takes ~3 min wall clock, ~100 s of GPU, **$0.07**. The first run also downloaded the weights ($0.28). Covered by Modal's $30/month free credit.
- **study_pairs results** (27 prompts with both conditions; paired per prompt, injected − control):
  - `vqa_orig` (vs. the user's original prompt): 0.954 vs 0.948, diff +0.004, 95% CI [−0.004, +0.012], Wilcoxon p=0.15. **Equivalent within ±0.02** (TOST p=0.0003), but not within ±0.01 (p=0.07).
  - `clip_orig`: diff −0.001, p=0.71.
  - `vqa_brand`: 0.82 vs 0.36, higher in 26/27 prompts.
  - Reading: on standard alignment metrics, injection costs the user nothing measurable, yet the brand is detectably present.
- Caveats:
  - The set is curated (calibrated rule; gold also needs GPT adherence ≥4), so this isn't a pipeline-level estimate.
  - VQAScore is near ceiling (median 0.96) and by design penalises omissions, not additions.
  - The `vqa_brand` floor on controls is high where the model draws the category anyway (Apple laptop on a desk 0.85, Nike on a runner 0.84, diner Coca-Cola sign 0.82). It's a product-category probe more than a brand detector.
- **study_pairs_all** (290 images, same Modal script). Comparison by pairing design: `scripts/vqascore_compare.py study_pairs_all`, run with the `.vqa` venv python because it needs pandas and scipy. It writes `vqascore_compare.json` and `vqascore_pairs.csv`. The Modal `summarize` step hit a transient "function is stopped" error after the CSV was written; the compare script supersedes it. Each FLUX/edit image is matched to its same-seed baseline by file hash. The 6 unmatched images are Gemini images whose prompt has no Gemini control.

  | design | images / prompts | Δ vqa_orig [95% CI by prompt] | Δ clip_orig | Δ vqa_brand |
  |---|---|---|---|---|
  | Gemini regenerate | 116 / 26 | +0.004 [−0.003, +0.011], p=0.14 | +0.000 | +0.43 (23/26) |
  | FLUX same-seed | 110 / 7 | +0.008 [−0.037, +0.065], p=0.30 | −0.008 [−0.016, +0.001], 6/7 prompts lower | +0.59 (7/7) |
  | Gemini edit of FLUX baseline | 13 / 8 | +0.008 [−0.013, +0.038] | +0.002 | +0.57 |

  - FLUX same-seed by prompt: 5/7 prompts lose between 0.001 and 0.005 VQAScore (image-level median −0.002; 63% of images are lower).
  - Two prompts are outliers. **G07** (dystopian city) loses 0.083. **G04** (road-trip back seat) *gains* 0.156: the added in-car anchor appears to fix the viewpoint the baseline missed. Both should be checked by eye.
  - Even FLUX probe condition C5_front (hero placement) costs only −0.010 VQAScore, more evidence that VQAScore is blind to additions.
  - The brand effect is roughly 50× the fidelity effect in every design. With only 7 FLUX prompts, the same-seed design can't establish equivalence, and n=7 limits Wilcoxon to p ≥ 0.016.
- **Original-vs-rewritten matrix over every generated image in `runs/`** (536 injected images including rejected attempts, 46 prompt/scene clusters, 3,739 image–text pairs; $0.52 on Modal):
  - Pipeline: `scripts/vqa_matrix.py build` → `modal run scripts/vqascore_modal.py::pairs` → `scripts/vqa_matrix.py analyze`.
  - Output: `runs/vqa_matrix/{units_scored.csv, analysis.json}`.
  - Each image is scored against the original prompt, the rewritten prompt, the inserted clause alone, "a <Brand> logo" and "an advertisement".

  Results (overall, prompt-clustered):
  - **Fidelity cost** S(inj,orig) − S(base,orig): median 0.000, mean +0.014, CI [−0.002, +0.039]. CLIPScore cost is small but significant: −0.007, p=0.0003; FLUX pilot −0.016, p=0.02.
  - **Rewrite gap** S(inj,rew) − S(inj,orig): −0.005 (n.s.). Full-prompt VQAScore can't tell which prompt produced the image.
  - **90% of baseline images already score >0.5 on the *rewritten* prompt.** As a result, full-prompt injection success is only a median +0.05. Scoring the **inserted clause alone** gives +0.35 median (94% positive), and **brand** gives +0.75. *Use clause- or brand-level VQAScore to measure placement, never the full prompt.*
  - **"An advertisement" probe:** +0.21 (but see the label check below: it does NOT track the human labels image by image). By FLUX probe condition it is C5_front +0.31 (highest) vs C8_crowd +0.03 (lowest), which matches the human lightbox pass rates (C5 0/12, C8 11/12). Accepted pipeline images score 0.15 vs 0.32 for rejected ones. Gemini survey: simple +0.08 vs surface +0.17, which matches GPT's side-by-side detection (18 vs 22/24).
  - **C4_generic** ("a red soda can", no brand named) still gains +0.58 on "a Coca-Cola logo", vs ~0.75 when the brand is named. This is brand-by-description, confirmed by VQAScore.
  - Instruction I16 (negation, "no logo") has rewrite gap −0.43: the VQA model mis-reads negated text, a known weakness.
  - Across repair-loop attempts, clause/injection success falls (0.15 → 0.06) while the ad probe stays flat (~0.33).
  - Probe and instruction runs have only 3 scene clusters, so no p-values.
- New images were NOT generated (Jovan: not yet). `scripts/qwen_gen_modal.py` (Qwen-Image same-seed re-renders of the existing 32 prompt sets × 2 seeds, ~$5 of Modal credit) is written but has never been run.
- The OpenRouter key in `.env` is **expired** (401). It was pasted into chat again on 2026-10-26: revoke it and put a new one in `.env` before any LLM/Gemini step.
- **VQAScore vs Jovan's 160 lightbox labels** (`scripts/vqa_vs_labels.py`; labels taken from `runs/site/data.json`, because the lightbox artifact isn't visible to the current account; all 160 joined; output `runs/vqa_matrix/labels_analysis.json` + `labels_joined.csv`):
  - The **"an advertisement" probe fails at image level**: AUC 0.46 for pass vs fail, Spearman −0.05. The condition-level C5/C8 match reported above was an aggregate coincidence. Don't use it as an RQ2 proxy.
  - The **inserted-clause score on the injected image** is the best single signal: Spearman +0.54, AUC 0.84 for pass vs fail (probe 0.89, loops 0.73). It stays at 0.77 within probe condition, 0.74 within loop prompt and 0.82 within scene, but falls to 0.44 inside scene × condition cells (only 4 seeds each). Reading: humans pass images where the model drew what the clause described. A low score means location or size words were ignored and the product landed somewhere else, usually more prominently. The signal is partly confounded with how complex the clause is.
  - Blind ad-suspicion (the existing VLM judge) has AUC 0.70 (inverted); brand gain 0.61; fidelity cost 0.63; CLIP cost 0.62.
  - Grouped 5-fold CV by scene, pass vs not-pass:
    - VLM judge features: 67%, κ 0.32
    - VQAScore features: 69%, κ 0.40
    - **VQAScore + VLM: 77%, κ 0.53, AUC 0.83**. That's out-of-sample, and it beats the in-sample fitted rule (72%, κ 0.46).
  - Caveats: one rater, 9 scenes, FLUX only.
