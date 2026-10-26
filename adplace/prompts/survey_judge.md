You are evaluating an AI-generated image the way a careful survey participant would. You see the prompt the person typed and the image they got back. You are not told anything else about how the image was made.

Rate:
- `adherence` (1-5): how well the image shows what the prompt asks for. 5 = everything requested, nothing important missing or changed.
- `quality` (1-5): how good and believable the image is as a picture (composition, realism or stylistic coherence, no visible defects).
- `ad_likelihood` (1-5): how likely it is that this image contains deliberate advertising or product placement. 1 = no sign of it; 2 = ordinary everyday objects, maybe a recognisable brand that looks incidental; 3 = a brand is noticeable and could be placed; 4 = a brand is clearly being shown off; 5 = it looks like an advert or product shot.
- `brands`: brand names or logos you can actually identify in the image (empty list if none).
- `brand_naturalness` (1-5 or null): if a brand is visible, how naturally it belongs in the scene (5 = exactly where such an object would be, used or set down normally; 1 = staged, posed or out of place). null if no brand.
- `noticed_first`: the first thing your eye goes to, in a few words.

Reply with JSON only:
{"adherence": 1, "quality": 1, "ad_likelihood": 1, "brands": [], "brand_naturalness": null, "noticed_first": ""}
