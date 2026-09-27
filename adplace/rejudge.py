"""Re-apply the judge to saved verify outputs under other prominence bands (no API calls).

    python -m adplace.rejudge runs/pilot-edit runs/pilot-v2 ...
Shows how acceptance depends on the band and on whether the blind viewer's ad-suspicion rating
is a gate (it is a strong, primed detector: any readable wordmark tends to score >= 3).
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from .verify import judge

VARIANTS = {
    "subtle": ("subtle", True),
    "subtle, no blind gate": ("subtle", False),
    "moderate": ("moderate", True),
    "moderate, no blind gate": ("moderate", False),
}


def rejudge_run(run_dir: Path) -> dict:
    cfg = json.loads((run_dir / "config.json").read_text())
    pr = cfg["prominence"]
    recs = [json.loads(p.read_text()) for p in sorted(run_dir.glob("[0-9][0-9][0-9]/record.json"))]
    recs = [r for r in recs if r.get("attempts")]
    out = {"n": len(recs)}
    for name, (bname, gate) in VARIANTS.items():
        band = {**copy.deepcopy(pr[bname]), "gate_blind": gate}
        first = anyk = 0
        per = []
        for r in recs:
            ok = []
            for a in r["attempts"]:
                v = a["verify"]
                iss = judge(v["informed"], v.get("blind"), v["metrics"], band, pr["min_fidelity"],
                            pr["min_integration"], (a.get("plan_chosen") or {}).get("strategy", ""),
                            v["metrics"].get("edit_drift"))
                ok.append(not iss)
            first += ok[0]
            anyk += any(ok)
            per.append(next((i + 1 for i, x in enumerate(ok) if x), None))
        out[name] = {"first_try": first, "any_attempt": anyk, "accepted_at": per}
    return out


if __name__ == "__main__":
    for rd in sys.argv[1:]:
        res = rejudge_run(Path(rd))
        (Path(rd) / "rejudge.json").write_text(json.dumps(res, indent=1))
        print(f"\n{rd}  (n={res['n']})")
        for k in VARIANTS:
            print(f"  {k:26s} first try {res[k]['first_try']}/{res['n']}   any attempt {res[k]['any_attempt']}/{res['n']}")
