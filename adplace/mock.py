"""Offline stand-ins for OpenRouter and the image backends, for tests and `--mock` runs.

MockLLM returns canned JSON keyed on `task`. The informed-verify replies come from a script
(a list consumed in order), so a test can drive the loop through any repair branch:
    MockLLM(verify_script=["TOO_PROMINENT", "PASS"])
MockImage draws a flat scene with a red rectangle standing in for the product.
"""
from __future__ import annotations

import io
import itertools
import json

from .llm import Ledger

# informed-verify replies by scenario name
SCENARIOS = {
    "PASS": {"product_found": True, "product_boxes": [[700, 700, 780, 740]], "main_subject_box": [300, 100, 800, 500],
             "legibility": 2, "draws_attention": 2, "integration": 5, "fidelity_to_original": 5,
             "wordmark_correct": True, "lookalikes": 0, "artifact_severity": 0},
    "TOO_PROMINENT": {"product_found": True, "product_boxes": [[300, 350, 700, 650]], "main_subject_box": [300, 100, 800, 500],
                      "legibility": 4, "draws_attention": 5, "integration": 4, "fidelity_to_original": 5,
                      "wordmark_correct": True, "artifact_severity": 0,
                      "fix_suggestion": "shrink the can and move it behind the basket"},
    "ABSENT": {"product_found": False, "product_boxes": [], "main_subject_box": [300, 100, 800, 500],
               "legibility": 0, "draws_attention": 1, "integration": 1, "fidelity_to_original": 5, "artifact_severity": 0},
    "DUPLICATES": {"product_found": True, "product_boxes": [[700, 700, 780, 740], [700, 100, 780, 140], [100, 800, 180, 840]],
                   "main_subject_box": [300, 100, 800, 500], "legibility": 2, "draws_attention": 2,
                   "integration": 5, "fidelity_to_original": 5, "artifact_severity": 0},
    "FIDELITY_LOSS": {"product_found": True, "product_boxes": [[700, 700, 780, 740]], "main_subject_box": [300, 100, 800, 500],
                      "legibility": 2, "draws_attention": 2, "integration": 5, "fidelity_to_original": 2,
                      "missing_or_altered": ["the dog", "the kite"], "artifact_severity": 0},
    "ARTIFACTS": {"product_found": True, "product_boxes": [[700, 700, 780, 740]], "main_subject_box": [300, 100, 800, 500],
                  "legibility": 2, "draws_attention": 2, "integration": 5, "fidelity_to_original": 5,
                  "artifacts": ["melted hands"], "artifact_severity": 3},
    "ON_SUBJECT": {"product_found": True, "product_boxes": [[400, 200, 460, 240]], "main_subject_box": [350, 150, 600, 400],
                   "legibility": 2, "draws_attention": 2, "integration": 5, "fidelity_to_original": 5, "artifact_severity": 0},
}


class MockLLM:
    def __init__(self, verify_script=("TOO_PROMINENT", "PASS"), decision="INJECT", ledger: Ledger | None = None,
                 blind_suspicion: int = 1):
        self.script = itertools.chain(verify_script, itertools.repeat(verify_script[-1]))
        self.decision = decision
        self.blind_suspicion = blind_suspicion
        self.calls: list[str] = []
        self.ledger = ledger or _MemLedger()

    def resolve(self, wanted, **_):
        return wanted if isinstance(wanted, str) else wanted[0]

    def chat_json(self, model, system, user, task, temperature=0.2, images=None):
        self.calls.append(task)
        self.ledger.record(task, "mock", cost=0.0001)
        u = user if isinstance(user, dict) else {}
        if task == "classify":
            return {"decision": self.decision, "reason": "mock", "exclude_flags": [], "prompt_specificity": "medium",
                    "candidate_surfaces": ["table"],
                    "brands": [{"brand_key": "coca-cola", "fit": 8, "why": "mock"}] if self.decision == "INJECT" else []}
        if task == "plan":
            cand = lambda i, strat, anchor, sc: {"id": i, "strategy": strat, "form": "red aluminium can", "anchor": anchor,
                                                 "spatial_relation": f"on the {anchor}", "depth_plane": "midground",
                                                 "frame_region": "lower right third", "scale_hint": "a third of the basket",
                                                 "orientation": "turned away", "lighting": "same light",
                                                 "scores": {"congruence": sc, "subtlety": sc, "recognisability": 4,
                                                            "preservation": sc, "renderability": 4}}
            return {"scene": {"focal_subject": "family", "protected_elements": ["family", "beach"]},
                    "candidates": [cand(1, "attach_to_surface", "blanket", 5), cand(2, "substitute_implied", "cooler", 4),
                                   cand(3, "environmental", "kiosk", 3)], "chosen_id": 1, "why": "mock"}
        if task.startswith("rewrite"):
            base = u.get("previous_rewrite") or u.get("original_prompt", "")
            if u.get("previous_rewrite"):
                return {"rewritten_prompt": base, "inserted_clause": "", "notes": "mock revision"}
            return {"rewritten_prompt": f"{base}, a single red Coca-Cola can resting on the blanket behind the basket, turned away",
                    "inserted_clause": "a single red Coca-Cola can resting on the blanket behind the basket, turned away"}
        if task == "verify_blind":
            return {"description": "mock scene", "out_of_place": [], "brands_seen": [], "ad_suspicion": self.blind_suspicion}
        if task == "verify_informed" and "BEFORE" in str(u.get("note", "")):
            return json.loads(json.dumps(SCENARIOS["ABSENT"]))          # baseline scan: no brand present
        if task == "verify_informed":
            return json.loads(json.dumps(SCENARIOS[next(self.script)]))
        if task == "vpp_plan":
            return {"anchor": "the basket", "form": "red aluminium can",
                    "instruction": "Add one small Coca-Cola can half hidden behind the basket.", "rationale": "mock"}
        if task == "repair":
            return {"edit_instruction": "Make the can smaller and move it behind the basket.",
                    "revision_feedback": "Make the can smaller and move it further back.", "rationale": "mock"}
        raise ValueError(f"MockLLM: unknown task {task}")


class _MemLedger:
    def __init__(self):
        self.spent, self.images, self.rows = 0.0, 0, []

    def check(self, needs_image=False):
        pass

    def begin(self, tag=""):
        pass

    def thread_spent(self):
        return self.spent

    def record(self, task, model, cost=0.0, images=0, **extra):
        self.spent += cost
        self.images += images
        self.rows.append((task, model, cost, images))


class MockImage:
    """Draws a flat scene; the 'product' is a red box whose size shrinks after each edit."""

    def __init__(self, ledger=None):
        self.ledger = ledger
        self.calls: list[tuple[str, str]] = []

    def _draw(self, scale: float) -> bytes:
        from PIL import Image, ImageDraw
        im = Image.new("RGB", (256, 256), (135, 180, 220))
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 170, 256, 256], fill=(220, 200, 150))           # sand
        dr.ellipse([60, 110, 120, 200], fill=(90, 60, 40))                # "family"
        s = int(40 * scale)
        dr.rectangle([180, 190 - s, 180 + s // 2, 190], fill=(200, 20, 20))  # "product"
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    def generate(self, prompt, seed, width, height):
        self.calls.append(("generate", prompt))
        if self.ledger:
            self.ledger.record("mock_generate", "mock", images=1)
        return self._draw(1.0)

    def edit(self, image, instruction, seed):
        self.calls.append(("edit", instruction))
        if self.ledger:
            self.ledger.record("mock_edit", "mock", images=1)
        return self._draw(0.5)
