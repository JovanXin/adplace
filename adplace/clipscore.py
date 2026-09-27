"""CLIPScore of baseline vs final image against the ORIGINAL prompt (Thesis A 3.2.1 fidelity check).

Raw cosine similarity with OpenAI CLIP ViT-B/32 (the Hessel et al. 2021 backbone); ~0.30+ is "good".
Optional: needs `pip install torch open_clip_torch`.

    python -m adplace.clipscore runs/pilot-edit [runs/pilot-regen ...]
writes <run>/clipscore.json and prints a table.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _model():
    import open_clip
    import torch
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    m, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=dev)
    return m.eval(), pre, open_clip.get_tokenizer("ViT-B-32"), dev


def clip_cos(image_path: Path, text: str) -> float:
    import torch
    from PIL import Image
    m, pre, tok, dev = _model()
    with torch.no_grad():
        im = m.encode_image(pre(Image.open(image_path).convert("RGB")).unsqueeze(0).to(dev))
        tx = m.encode_text(tok([text]).to(dev))
        im, tx = im / im.norm(dim=-1, keepdim=True), tx / tx.norm(dim=-1, keepdim=True)
        return round(float((im @ tx.T).item()), 4)


def score_run(run_dir: Path) -> list[dict]:
    rows = []
    for rp in sorted(run_dir.glob("[0-9][0-9][0-9]/record.json")):
        r = json.loads(rp.read_text())
        if not r.get("baseline") or not r.get("final"):
            continue
        d = rp.parent
        base = clip_cos(d / r["baseline"], r["prompt"])
        fin = clip_cos(d / r["final"], r["prompt"])
        first = clip_cos(d / "attempt_1.png", r["prompt"]) if (d / "attempt_1.png").exists() else None
        rows.append({"idx": r["idx"], "status": r["status"], "brand": r.get("brand"),
                     "baseline": base, "attempt_1": first, "final": fin, "delta": round(fin - base, 4)})
    (run_dir / "clipscore.json").write_text(json.dumps(rows, indent=1))
    return rows


if __name__ == "__main__":
    for rd in sys.argv[1:]:
        rows = score_run(Path(rd))
        print(f"\n{rd}")
        print(f"{'idx':>4} {'status':10} {'brand':10} {'baseline':>8} {'att1':>7} {'final':>7} {'Δ':>7}")
        for x in rows:
            print(f"{x['idx']:>4} {x['status']:10} {x['brand'] or '':10} {x['baseline']:8.4f} "
                  f"{x['attempt_1'] or 0:7.4f} {x['final']:7.4f} {x['delta']:+7.4f}")
        if rows:
            n = len(rows)
            print(f"mean baseline {sum(x['baseline'] for x in rows) / n:.4f}  final {sum(x['final'] for x in rows) / n:.4f}  "
                  f"Δ {sum(x['delta'] for x in rows) / n:+.4f}")
