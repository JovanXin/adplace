You are checking whether a text-to-image model (FLUX.2) followed one specific placement instruction. You get two images made with the same random seed:
- **Image A** (first): the reference, from the user's original prompt.
- **Image B** (second): from the same prompt with one instruction appended.

Be literal and strict. Measure what is in image B; do not give credit for effort.

1. Split the instruction into its **atomic requirements** (e.g. "a Coca-Cola bottle half hidden inside the picnic basket" → "a Coca-Cola bottle is present", "exactly one", "it is inside the basket", "it is about half hidden"). Keep each requirement short. Always include "the product is present" and, if a count is implied ("a", "one", "two"), the count.
2. Judge each requirement in image B: "yes" | "partly" | "no", with a few words of evidence.
3. `overall`: "yes" only if every requirement is yes; "no" if any core requirement (presence, the relation/position/state that the instruction is about) is no; else "partly".
4. `what_it_did_instead`: if not fully followed, one sentence on what FLUX drew instead (e.g. "put the bottle standing on the blanket in front of the basket, fully visible").
5. `count`: number of instances of the branded product in B.
6. `scene_changed` (1-5): how much the rest of the picture changed versus A (1 = identical apart from the product, 5 = a different picture).
7. `prominence` (1-5): how much the product pulls the eye in B.

Reply with JSON only:
{"requirements": [{"req": "", "met": "yes", "evidence": ""}], "overall": "yes", "what_it_did_instead": "", "count": 1, "scene_changed": 1, "prominence": 1}
