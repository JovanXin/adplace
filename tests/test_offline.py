"""Offline tests: no network. Run with  .venv/bin/python -m pytest -q  (or python tests/test_offline.py)."""
from __future__ import annotations

import copy
import io
import json
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adplace.loop import Ctx, choose_action, process_prompt, run  # noqa: E402
from adplace.mock import MockImage, MockLLM  # noqa: E402
from adplace.pipeline import keyword_gate, lint, load_catalog, score_candidate  # noqa: E402
from adplace.verify import box_metrics, edit_drift, judge  # noqa: E402

CFG = yaml.safe_load((ROOT / "config.example.yaml").read_text())
CFG["prominence"]["visibility"] = "subtle"   # the loop tests below were written against the subtle band
BAND = CFG["prominence"]["subtle"]


# ---------------------------------------------------------------- gate
def test_gate():
    assert keyword_gate("kids playing at a playground")["exclude_flags"] == ["minors"]
    assert "religion" in keyword_gate("interior of a gothic church, light rays")["exclude_flags"]
    assert "existing_brand" in keyword_gate("a man holding a Pepsi")["exclude_flags"]
    assert keyword_gate("a farmer harvesting ripe red apples in an orchard") is None      # fruit, not brand
    assert keyword_gate("an old apple macbook on a desk") is not None
    assert keyword_gate("friends at a bbq in the backyard") is None


# ---------------------------------------------------------------- lint
O = "A vintage-style painting of a family having a picnic on the beach"


def test_lint_ok():
    r = lint(O, O + ", a small painted Coca-Cola glass bottle half hidden behind the wicker basket", "Coca-Cola")
    assert r.ok, r.problems
    assert r.inserted.startswith("a small painted Coca-Cola")


def test_lint_catches():
    assert any("negation" in p for p in lint(O, O + ", a Coca-Cola bottle not facing the viewer", "Coca-Cola").problems)
    assert any("brand appears at" in p for p in lint(O, "A Coca-Cola bottle and " + O, "Coca-Cola").problems)
    assert any("named 2 times" in p for p in lint(O, O + ", Coca-Cola bottle, Coca-Cola sign", "Coca-Cola").problems)
    assert any("plural" in p for p in lint(O, O + ", some Coca-Cola bottles", "Coca-Cola").problems)
    assert any("banned word" in p for p in lint(O, O + ", an iconic Coca-Cola bottle", "Coca-Cola").problems)
    assert any("kept" in p for p in lint(O, "a picnic, Coca-Cola bottle", "Coca-Cola").problems)


def test_planner_focal_penalty():
    base = {"strategy": "attach_to_surface", "frame_region": "right edge", "depth_plane": "midground",
            "scores": {"congruence": 5, "subtlety": 5, "recognisability": 4, "preservation": 5, "renderability": 5}}
    on_jacket = {**base, "anchor": "the puffer jacket sleeve"}
    on_rack = {**base, "anchor": "a clothes rack behind the model"}
    assert score_candidate(on_jacket, "puffer jacket artwork") < score_candidate(on_rack, "puffer jacket artwork") - 1.5


# ---------------------------------------------------------------- metrics + judge
def test_box_metrics():
    m = box_metrics([[450, 450, 550, 550]], [0, 0, 1000, 1000])
    assert m["count"] == 1 and abs(m["area"] - 0.01) < 1e-6 and m["centrality"] == 1.0
    corner = box_metrics([[900, 900, 1000, 1000]], None)
    assert corner["centrality"] < 0.2
    dup = box_metrics([[100, 100, 200, 200], [102, 101, 199, 201], [700, 700, 800, 800]], None)
    assert dup["count"] == 2          # near-identical boxes merged
    pre = box_metrics([[100, 100, 200, 200], [700, 700, 800, 800]], None, exclude_boxes=[[101, 99, 201, 199]])
    assert pre["count"] == 1 and pre["largest_box"][0] == 0.7   # baseline's own brand instance ignored


def _j(inf, blind=None, strategy="attach_to_surface", drift=None):
    m = box_metrics(inf.get("product_boxes"), inf.get("main_subject_box"))
    return [i["code"] for i in judge(inf, blind, m, BAND, 4, 4, strategy, drift)]


GOOD = {"product_found": True, "product_boxes": [[700, 700, 800, 760]], "main_subject_box": [300, 100, 800, 500],
        "legibility": 2, "draws_attention": 2, "integration": 5, "fidelity_to_original": 5, "artifact_severity": 0}


def test_judge_accepts_good():
    assert _j(GOOD) == []


def test_judge_codes():
    assert _j({**GOOD, "product_found": False, "product_boxes": []}) == ["ABSENT"]
    assert _j({**GOOD, "legibility": 0}) == ["ABSENT"]
    assert "TOO_PROMINENT" in _j({**GOOD, "product_boxes": [[300, 300, 700, 700]]})
    assert "TOO_PROMINENT" in _j({**GOOD, "legibility": 4})
    assert "TOO_PROMINENT" in _j(GOOD, blind={"ad_suspicion": 4})
    assert "DUPLICATES" in _j({**GOOD, "product_boxes": [[700, 700, 800, 760], [100, 100, 200, 160]]})
    assert "DUPLICATES" not in _j({**GOOD, "lookalikes": 14})          # scene clutter is not a display
    assert "FIDELITY_LOSS" in _j({**GOOD, "fidelity_to_original": 3})
    assert "FIDELITY_LOSS" in _j(GOOD, drift=0.2)
    assert "BAD_INTEGRATION" in _j({**GOOD, "integration": 2})
    assert "ARTIFACTS" in _j({**GOOD, "artifact_severity": 2})
    assert "GARBLED_BRAND" in _j({**GOOD, "wordmark_correct": False})
    on_subj = {**GOOD, "product_boxes": [[400, 200, 450, 240]], "main_subject_box": [350, 150, 600, 400]}
    assert "ON_SUBJECT" in _j(on_subj)
    assert "ON_SUBJECT" not in _j(on_subj, strategy="incidental_use")
    # a group scene spanning the frame is not a 'compact' subject
    assert "ON_SUBJECT" not in _j({**GOOD, "main_subject_box": [300, 50, 950, 950]})


def test_priority_order():
    codes = _j({**GOOD, "product_boxes": [[300, 300, 700, 700]], "fidelity_to_original": 2})
    assert codes[0] == "FIDELITY_LOSS"


def test_edit_drift():
    mi = MockImage()
    a, b = mi._draw(1.0), mi._draw(0.5)
    assert edit_drift(a, a, []) == 0.0
    assert edit_drift(a, b, [(0.5, 0.65, 0.8, 0.8)]) < 0.01      # change is inside the masked product box
    assert edit_drift(a, b, []) > 0.0


# ---------------------------------------------------------------- policy
def test_policy_escalates_to_replan():
    iss = [{"code": "TOO_PROMINENT", "detail": ""}]
    two = [{"codes": ["TOO_PROMINENT"], "action": "regenerate", "mode": "feedback", "targets": ["TOO_PROMINENT"]}] * 2
    assert choose_action(iss, two, False)["mode"] == "replan"


def test_lint_dilution_allowed():
    r = lint(O, "A vintage-style painting of a family laughing together having a picnic on the beach, "
                "a painted Coca-Cola bottle half hidden in the basket", "Coca-Cola", max_words=18)
    assert r.ok, r.problems


def test_policy():
    iss = lambda *cs: [{"code": c, "detail": ""} for c in cs]
    assert choose_action(iss("TOO_PROMINENT"), [], True)["action"] == "edit"
    assert choose_action(iss("TOO_PROMINENT"), [], False)["action"] == "regenerate"
    hist = [{"codes": ["TOO_PROMINENT"], "action": "edit", "targets": ["TOO_PROMINENT"]}]
    assert choose_action(iss("TOO_PROMINENT"), hist, True)["action"] == "regenerate"   # edit already failed
    assert choose_action(iss("FIDELITY_LOSS", "TOO_PROMINENT"), [], True)["mode"] == "replan"
    assert choose_action(iss("ARTIFACTS"), [], True)["mode"] == "reseed"
    assert choose_action(iss("ABSENT"), [], True)["action"] == "regenerate"
    two_absent = [{"codes": ["ABSENT"]}, {"codes": ["ABSENT"]}]
    assert choose_action(iss("ABSENT"), two_absent, True)["action"] == "edit"
    assert choose_action(iss("ABSENT"), two_absent, False)["action"] == "regenerate"


# ---------------------------------------------------------------- full mock loop
def _ctx(tmp, script, can_edit=True, decision="INJECT", mode="rewrite", max_attempts=4, **llm_kw):
    cfg = copy.deepcopy(CFG)
    cfg["generation"]["mode"] = mode
    cfg["loop"]["max_attempts"] = max_attempts
    llm = MockLLM(verify_script=script, decision=decision, **llm_kw)
    img = MockImage(llm.ledger)
    return Ctx(cfg=cfg, llm=llm, models={"classifier": "m", "rewriter": "m", "verifier": "m"}, gen=img,
               editor=img if can_edit else None, catalog=load_catalog(), run_dir=Path(tmp)), llm, img


def _run(script, **kw):
    with tempfile.TemporaryDirectory() as tmp:
        ctx, llm, img = _ctx(tmp, script, **kw)
        rec = process_prompt(ctx, 0, O)
        saved = json.loads((Path(tmp) / "000" / "record.json").read_text())
        assert saved["status"] == rec["status"]
        files = sorted(p.name for p in (Path(tmp) / "000").iterdir())
        return rec, llm, img, files


def test_loop_too_prominent_edit():
    rec, llm, img, files = _run(["TOO_PROMINENT", "PASS"])
    assert rec["status"] == "accepted" and len(rec["attempts"]) == 2
    assert [c[0] for c in img.calls] == ["generate", "generate", "edit"]      # baseline, attempt 1, edit
    assert rec["attempts"][0]["next_action"]["action"] == "edit"
    assert rec["attempts"][1]["verify"]["metrics"]["edit_drift"] is not None
    assert {"baseline.png", "attempt_1.png", "attempt_2.png", "record.json"} <= set(files)


def test_loop_regen_only():
    rec, llm, img, _ = _run(["TOO_PROMINENT", "PASS"], can_edit=False)
    assert rec["status"] == "accepted"
    assert [c[0] for c in img.calls] == ["generate", "generate", "generate"]
    assert "rewrite" in llm.calls and "repair" in llm.calls


def test_loop_absent_then_edit_insert():
    rec, _, img, _ = _run(["ABSENT", "ABSENT", "PASS"])
    assert rec["status"] == "accepted"
    assert [c[0] for c in img.calls] == ["generate", "generate", "generate", "edit"]


def test_loop_duplicates_and_on_subject():
    for s in ("DUPLICATES", "ON_SUBJECT"):
        rec, _, img, _ = _run([s, "PASS"])
        assert rec["status"] == "accepted" and img.calls[-1][0] == "edit", s


def test_loop_fidelity_replans():
    rec, llm, img, _ = _run(["FIDELITY_LOSS", "PASS"])
    assert rec["status"] == "accepted"
    assert rec["attempts"][1]["plan_chosen"]["id"] != rec["attempts"][0]["plan_chosen"]["id"]


def test_loop_artifacts_reseed():
    rec, _, _, _ = _run(["ARTIFACTS", "PASS"])
    assert rec["attempts"][1]["made_by"]["seed"] != rec["seed"]


def test_loop_gives_up_and_keeps_best():
    rec, _, _, _ = _run(["TOO_PROMINENT"], max_attempts=3)
    assert rec["status"] == "failed" and len(rec["attempts"]) == 3 and rec["final"]


def test_loop_blind_suspicion_blocks():
    rec, _, _, _ = _run(["PASS"], blind_suspicion=5, max_attempts=2)
    assert rec["status"] == "failed"
    assert "TOO_PROMINENT" in rec["attempts"][0]["codes"]


def test_loop_skip_and_vpp():
    rec, _, img, _ = _run(["PASS"], decision="SKIP")
    assert rec["status"] == "skip" and img.calls == []
    rec, _, img, _ = _run(["PASS"], mode="vpp")
    assert rec["status"] == "accepted" and [c[0] for c in img.calls] == ["generate", "edit"]


def test_run_summary_and_disclosure():
    with tempfile.TemporaryDirectory() as tmp:
        ctx, _, _ = _ctx(tmp, ["TOO_PROMINENT", "PASS"])
        s = run(ctx, [O, "friends at a bbq in the backyard"], workers=2)
        assert s["n_generated"] == 2 and s["issue_codes"]["TOO_PROMINENT"] >= 1
        rows = (Path(tmp) / "injections.csv").read_text().strip().splitlines()
        assert len(rows) == 3
        from adplace.report import build
        html = Path(build(Path(tmp))).read_text()
        assert "TOO_PROMINENT" in html and "Coca-Cola" in html


def test_reuse_paired():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        ctx, _, _ = _ctx(a, ["TOO_PROMINENT", "PASS"])
        process_prompt(ctx, 0, O)
        ctx2, llm2, img2 = _ctx(b, ["PASS"], can_edit=False)
        ctx2.reuse = Path(a)
        rec = process_prompt(ctx2, 0, O)
        assert rec["reused_from"] and "classify" not in llm2.calls and "plan" not in llm2.calls
        assert rec["attempts"][0]["codes"] == ["TOO_PROMINENT"]            # attempt 1 copied, not re-verified
        assert [c[0] for c in img2.calls] == ["generate"]                  # only the regen repair was generated
        assert rec["status"] == "accepted"


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f()
        print("ok", f.__name__)


def test_natural_band_logs_but_does_not_gate_blind_suspicion():
    band = CFG["prominence"]["natural"]
    inf = {"product_found": True, "legibility": 3, "draws_attention": 3, "fidelity_to_original": 5,
           "integration": 5, "artifact_severity": 0}
    m = box_metrics([[700, 700, 800, 760]], [0, 0, 1000, 1000])
    assert judge(inf, {"ad_suspicion": 5}, m, band, 4, 4) == []


def test_scene_loss_from_baseline_comparison():
    band = CFG["prominence"]["natural"]
    inf = {"product_found": True, "legibility": 2, "draws_attention": 2, "fidelity_to_original": 5,
           "integration": 5, "artifact_severity": 0,
           "reference_objects": ["desk lamp", "coffee mug", "open notebook", "papers"],
           "missing_vs_reference": ["desk lamp", "coffee mug", "papers"]}
    m = box_metrics([[700, 700, 800, 760]], [0, 0, 1000, 1000])
    assert [i["code"] for i in judge(inf, None, m, band, 4, 4)] == ["SCENE_LOSS"]
    inf["missing_vs_reference"] = ["papers"]
    assert judge(inf, None, m, band, 4, 4) == []
