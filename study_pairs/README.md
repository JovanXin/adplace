# Study text-image pairs (Gemini), built 2026-10-21

Every Gemini-rendered image in `runs/` that passes the pipeline's **calibrated** placement rule (the rule fitted to
Jovan's 160 lightbox labels: 72% agreement, kappa 0.46), plus the matching no-ad image for each prompt.
Rebuild with `.venv/bin/python scripts/collect_good_pairs.py`. Add `--all` to include the FLUX/Qwen images; that
version writes to `study_pairs_all/`. No API calls are made; it re-reads the stored verifier output.

## Counts
- 135 injected images: **33 gold** + 102 good
- 36 controls (no-inject images)
- 29 user prompts
- 11 brands: Coca-Cola 34, Apple 31, Nike 19, Starbucks 12, Pepsi 10, Red Bull 8, IKEA 7, Xbox 6, Adidas 3, Heineken 3, Samsung 2
- Sources: survey-closed 64, survey-surface 47, pilot-vpp-gemini 13, pilot-gemini-gen 11

## Tiers
| tier | meaning |
|---|---|
| gold | Passes the calibrated rule, **and** a fresh GPT viewer names the brand, rates "looks like an ad" at 3/5 or below, and rates prompt adherence at 4/5 or above. This is the strictest "subtle but recognisable" set. |
| good | Passes the calibrated rule. The brand is recognisable (legibility 1–3), it is not a hero or product shot, it is not "put right in front", Gemini's blind ad-suspicion is 4 or below, and the scene is preserved. |
| control | The no-inject render of the same prompt. It is dropped where GPT saw a brand in it (e.g. the Tokyo store showing a Lawson sign). |

The deliberately prominent "bad" condition is never exported.

## Files
| file | who sees it | contents |
|---|---|---|
| `images/I0001.jpg` … | participants | Neutral file names that don't reveal the condition |
| `items.csv` | participants / survey app | `item_id, image, prompt` (the prompt is the user's original, un-rewritten prompt) |
| `best.csv` | researcher | Ready-made balanced set: 27 prompts × {control, most subtle injected} (18 gold, 9 good). Each pair shares an aspect ratio. |
| `key.csv` / `key.json` | researcher only | Condition, tier, brand, exact prompt sent to the model, model, method, scores, source path |
| `gallery.html` | researcher | Contact sheet grouped by prompt (gold has a yellow border, good green, control grey) |
| `stats.json` | researcher | The counts above |

Keep `key.csv`, `best.csv` and `gallery.html` away from participants; use them for post-study disclosure.

## Caveats
- **Closed-model images:** Gemini re-draws the scene whenever the prompt changes, so a control and an injected image
  are different compositions of the same prompt, not same-seed pairs.
- **Aspect ratio:** pilot-gemini-gen images (G01, G02, G05, G07) are 16:9 (1408×768). Everything else is 1:1.
  Don't mix the two within one comparison, because the shape would reveal the condition.
- **pilot-vpp-gemini:** these are Gemini *edits* of a FLUX baseline, so their control is the FLUX render.
- **Tier inputs:** the gold criterion uses the GPT judge, so don't also use that judge as an independent detectability measure on the gold set.
