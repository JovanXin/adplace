"""Does `seed` give closed image models a paired (same-composition) design?

For each model: A = orig@s, B = orig@s (repeat), C = orig + placement@s (small prompt change), D = orig@s+1.
Similarity = Pearson correlation of 64x64 greyscale thumbnails (1 = identical layout).
    .venv/bin/python scripts/closed_seed_test.py
"""
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from adplace.__main__ import load_cfg, make_llm  # noqa: E402
from adplace.llm import load_env  # noqa: E402

MODELS = ["google/gemini-3.1-flash-lite-image", "google/gemini-3.1-flash-image",
          "google/gemini-2.5-flash-image", "openai/gpt-5-image-mini"]
ORIG = "two friends having lunch at an outdoor cafe in Paris, film photo"
PLACED = ORIG + ", with plates, glasses and a Coca-Cola bottle on their table"
OUT = ROOT / "runs" / "closed-seed-test"


def sim(a: bytes, b: bytes) -> float:
    from PIL import Image
    import statistics as st
    A = list(Image.open(io.BytesIO(a)).convert("L").resize((64, 64)).getdata())
    B = list(Image.open(io.BytesIO(b)).convert("L").resize((64, 64)).getdata())
    ma, mb = st.mean(A), st.mean(B)
    num = sum((x - ma) * (y - mb) for x, y in zip(A, B))
    den = (sum((x - ma) ** 2 for x in A) * sum((y - mb) ** 2 for y in B)) ** 0.5
    return round(num / den, 3) if den else 0.0


def main():
    load_env(ROOT / ".env")
    llm = make_llm(load_cfg(None), False, run_id="closed-seed-test")
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(m, k, p, s) for m in MODELS for k, p, s in
            (("A", ORIG, 7), ("B", ORIG, 7), ("C", PLACED, 7), ("D", ORIG, 8))]

    def one(j):
        m, k, p, s = j
        t, s0 = time.time(), llm.ledger.thread_spent() if hasattr(llm.ledger, "thread_spent") else 0
        llm.ledger.begin(m)
        try:
            img = llm.image(m, p, task="seedtest", seed=s)
        except Exception as e:  # noqa: BLE001
            return j, None, str(e)[:200], 0, 0
        (OUT / f"{m.split('/')[1]}_{k}.png").write_bytes(img)
        return j, img, None, round(time.time() - t, 1), llm.ledger.thread_spent()

    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(one, jobs))
    by = {}
    for (m, k, _, _), img, err, sec, cost in res:
        by.setdefault(m, {})[k] = {"img": img, "err": err, "sec": sec, "cost": cost}
    report = {}
    for m, d in by.items():
        r = {k: {"sec": v["sec"], "cost": round(v["cost"], 4), "err": v["err"]} for k, v in d.items()}
        if all(d[k]["img"] for k in "ABCD"):
            r["repeat_AB"] = sim(d["A"]["img"], d["B"]["img"])
            r["placed_AC"] = sim(d["A"]["img"], d["C"]["img"])
            r["otherseed_AD"] = sim(d["A"]["img"], d["D"]["img"])
        report[m] = r
        print(m, {k: v for k, v in r.items() if k.endswith(("AB", "AC", "AD"))},
              "cost/img", round(sum(v["cost"] for v in d.values()) / 4, 4), "sec", [v["sec"] for v in d.values()],
              [v["err"] for v in d.values() if v["err"]])
    (OUT / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
