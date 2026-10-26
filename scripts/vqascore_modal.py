"""VQAScore (Lin et al. 2024, CLIP-FlanT5-XXL via t2v_metrics 1.2) + CLIPScore for a study-pairs export,
run on a Modal GPU so nothing heavy runs on the laptop.

One-time setup:
    uv tool install --python 3.12 modal
    modal setup                       # browser login, writes ~/.modal.toml

Run (from adplace/):
    modal run scripts/vqascore_modal.py                                   # study_pairs/, clip-flant5-xxl
    modal run scripts/vqascore_modal.py --pairs study_pairs_all
    modal run scripts/vqascore_modal.py --limit 6                         # smoke test
    modal run scripts/vqascore_modal.py --model clip-flant5-xl            # smaller model, cheaper GPU is fine

Every image gets three VQAScores, P("Yes" | image, 'Does this figure show "{text}"? ...'):
    vqa_orig   text = the user's ORIGINAL prompt     -> fidelity to what the user asked for (primary metric)
    vqa_sent   text = the prompt sent to the model   -> did the model follow the rewrite (= vqa_orig for controls)
    vqa_brand  text = "<brand> <product>"            -> brand presence; controls use their group's brand,
                                                        so they give the "brand there by chance" floor
plus clip_orig: raw cosine of OpenAI CLIP ViT-B/32 against the original prompt (same as adplace.clipscore).

Writes <pairs>/vqascore_<model>.csv (one row per image) and <pairs>/vqascore_<model>_summary.json.
Weights (~23 GB for xxl) are cached in the Modal Volume "adplace-hf-cache"; only the first run downloads them.
"""
import csv
import json
from collections import Counter
from pathlib import Path

import modal

app = modal.App("adplace-vqascore")
hf_cache = modal.Volume.from_name("adplace-hf-cache", create_if_missing=True)

# t2v_metrics 1.2 is the last release with CLIP-FlanT5 (3.0 dropped it). Its declared deps don't resolve any more,
# so it is installed --no-deps on top of the versions it was written against (checked: `import t2v_metrics` works).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("setuptools==69.5.1", "wheel")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", "numpy==1.26.4",
        "transformers==4.36.1", "tokenizers==0.15.2", "huggingface-hub==0.23.4", "accelerate==1.15.0",
        "diffusers==0.27.2", "datasets==2.19.2", "pyarrow==15.0.2", "fsspec==2024.3.1",
        "timm==0.6.13", "fairscale==0.4.13", "open-clip-torch==2.24.0", "sentencepiece", "ftfy", "regex",
        "tqdm", "iopath", "omegaconf", "scikit-learn", "scipy", "pandas<3", "openai", "tiktoken",
        "matplotlib", "opencv-python-headless==4.11.0.86", "protobuf", "pillow<11", "safetensors",
    )
    .pip_install("git+https://github.com/openai/CLIP.git", extra_options="--no-build-isolation")
    .pip_install("t2v-metrics==1.2", "image-reward==1.5", extra_options="--no-deps")
    .env({"HF_HOME": "/cache/hf"})
)

N_TEXTS = 3  # orig, sent, brand -- batch_forward needs the same number of texts per sample


@app.cls(image=image, gpu="L40S", memory=65536, volumes={"/cache": hf_cache}, timeout=3600)
class Scorer:
    model: str = modal.parameter(default="clip-flant5-xxl")

    @modal.enter()
    def load(self):
        import open_clip
        import t2v_metrics
        self.vqa = t2v_metrics.VQAScore(model=self.model, device="cuda", cache_dir="/cache/hf/hub")
        self.clip, _, self.pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai",
                                                                        device="cuda")
        self.clip.eval()
        self.tok = open_clip.get_tokenizer("ViT-B-32")
        hf_cache.commit()

    @modal.method()
    def score(self, items: list[dict], batch_size: int = 8) -> list[dict]:
        import torch
        from PIL import Image
        tmp = Path("/tmp/imgs")
        tmp.mkdir(exist_ok=True)
        for it in items:
            (tmp / it["file"]).write_bytes(it["jpg"])
        dataset = [{"images": [str(tmp / it["file"])], "texts": it["texts"]} for it in items]
        s = self.vqa.batch_forward(dataset=dataset, batch_size=batch_size)  # n x 1 image x 3 texts
        out = []
        with torch.no_grad():
            for it, row in zip(items, s[:, 0, :].tolist()):
                im = self.clip.encode_image(self.pre(Image.open(tmp / it["file"]).convert("RGB"))
                                            .unsqueeze(0).to("cuda"))
                tx = self.clip.encode_text(self.tok([it["texts"][0]]).to("cuda"))
                im, tx = im / im.norm(dim=-1, keepdim=True), tx / tx.norm(dim=-1, keepdim=True)
                out.append({"item_id": it["item_id"], "vqa_orig": round(row[0], 4), "vqa_sent": round(row[1], 4),
                            "vqa_brand": round(row[2], 4), "clip_orig": round(float((im @ tx.T).item()), 4)})
        return out


    @modal.method()
    def score_pairs(self, images: dict[str, bytes], pairs: list[list[str]], batch_size: int = 16) -> list[dict]:
        """Arbitrary (image name, text) pairs -> VQAScore + CLIP cosine; used by scripts/vqa_matrix.py."""
        import torch
        from PIL import Image
        tmp = Path("/tmp/pairs")
        tmp.mkdir(exist_ok=True)
        for name, b in images.items():
            (tmp / f"{name}.jpg").write_bytes(b)
        dataset = [{"images": [str(tmp / f"{h}.jpg")], "texts": [t]} for h, t in pairs]
        vqa = self.vqa.batch_forward(dataset=dataset, batch_size=batch_size)[:, 0, 0].tolist()
        with torch.no_grad():
            emb = {}
            for h in {h for h, _ in pairs}:
                e = self.clip.encode_image(self.pre(Image.open(tmp / f"{h}.jpg").convert("RGB")).unsqueeze(0).to("cuda"))
                emb[h] = e / e.norm(dim=-1, keepdim=True)
            texts = sorted({t for _, t in pairs})
            temb = {}
            for i in range(0, len(texts), 64):
                e = self.clip.encode_text(self.tok(texts[i:i + 64]).to("cuda"))
                temb.update(zip(texts[i:i + 64], e / e.norm(dim=-1, keepdim=True)))
            clip = [float(emb[h][0] @ temb[t]) for h, t in pairs]
        return [{"h": h, "text": t, "vqa": round(v, 4), "clip": round(c, 4)} for (h, t), v, c in zip(pairs, vqa, clip)]


@app.local_entrypoint()
def pairs(manifest_dir: str = "runs/vqa_matrix", model: str = "clip-flant5-xxl", chunk: int = 100):
    """Score <manifest_dir>/pairs.json (built by scripts/vqa_matrix.py build) -> <manifest_dir>/scores.json."""
    d = Path(manifest_dir)
    todo = json.loads((d / "pairs.json").read_text())
    names = sorted({h for h, _ in todo})
    scorer, out = Scorer(model=model), []
    for i in range(0, len(names), chunk):
        part = set(names[i:i + chunk])
        imgs = {h: (d / "img" / f"{h}.jpg").read_bytes() for h in part}
        ps = [p for p in todo if p[0] in part]
        out += scorer.score_pairs.remote(imgs, ps)
        print(f"  {min(i + chunk, len(names))}/{len(names)} images, {len(out)}/{len(todo)} pairs scored")
        (d / "scores.json").write_text(json.dumps(out))
    print(f"wrote {d / 'scores.json'}")


@app.function(image=image)
def summarize(rows: list[dict]) -> dict:
    """Per-condition/tier means, plus a per-prompt paired test: for each group with both a control and an
    injected image, diff = mean(injected) - mean(control); Wilcoxon signed-rank over groups."""
    import numpy as np
    from scipy.stats import wilcoxon
    metrics = ["vqa_orig", "vqa_sent", "vqa_brand", "clip_orig"]

    def desc(rs):
        return {"n": len(rs), **{m: [round(float(np.mean([r[m] for r in rs])), 4),
                                     round(float(np.std([r[m] for r in rs])), 4)] for m in metrics}}

    res = {"by_condition": {}, "by_tier": {}, "paired_by_prompt": {}}
    for key, field in (("by_condition", "condition"), ("by_tier", "tier")):
        for v in sorted({r[field] for r in rows}):
            res[key][v] = desc([r for r in rows if r[field] == v])
    groups = {}
    for r in rows:
        groups.setdefault(r["group_id"], {"control": [], "injected": []})[r["condition"]].append(r)
    paired = {g: v for g, v in groups.items() if v["control"] and v["injected"]}
    for m in metrics:
        d = [np.mean([r[m] for r in v["injected"]]) - np.mean([r[m] for r in v["control"]]) for v in paired.values()]
        p = float(wilcoxon(d).pvalue) if len(d) >= 6 and any(d) else None
        res["paired_by_prompt"][m] = {"n_prompts": len(d), "mean_diff": round(float(np.mean(d)), 4),
                                      "median_diff": round(float(np.median(d)), 4),
                                      "injected_higher": int(sum(x > 0 for x in d)),
                                      "wilcoxon_p": None if p is None else round(p, 4)}
    return res


def brand_text(r: dict) -> str:
    return r["product"] if r["brand"].lower() in r["product"].lower() else f"{r['brand']} {r['product']}"


@app.local_entrypoint()
def main(pairs: str = "study_pairs", model: str = "clip-flant5-xxl", limit: int = 0, batch_size: int = 8):
    d = Path(pairs)
    rows = list(csv.DictReader(open(d / "key.csv")))
    # controls have no brand: score them against their group's most common injected brand/product
    group_brand = {}
    for g in {r["group_id"] for r in rows}:
        c = Counter(brand_text(r) for r in rows if r["group_id"] == g and r["condition"] == "injected")
        group_brand[g] = c.most_common(1)[0][0] if c else ""
    if limit:
        rows = rows[:limit]
    items = [{"item_id": r["item_id"], "file": r["file"], "jpg": (d / "images" / r["file"]).read_bytes(),
              "texts": [r["user_prompt"], r["prompt_sent"],
                        brand_text(r) if r["condition"] == "injected" else group_brand[r["group_id"]]]}
             for r in rows]
    assert all(len(it["texts"]) == N_TEXTS and all(it["texts"]) for it in items)
    print(f"scoring {len(items)} images from {d} with {model} on Modal ...")
    scores = {s["item_id"]: s for s in Scorer(model=model).score.remote(items, batch_size)}

    keep = ["item_id", "group_id", "condition", "tier", "brand", "product", "model", "method", "run", "seed"]
    out = [{**{k: r[k] for k in keep}, "brand_text": it["texts"][2], **scores[r["item_id"]]}
           for r, it in zip(rows, items)]
    tag = model.replace("/", "_")
    with open(d / f"vqascore_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    summ = summarize.remote(out)
    (d / f"vqascore_{tag}_summary.json").write_text(json.dumps(summ, indent=1))

    print(f"\nwrote {d / f'vqascore_{tag}.csv'}\n")
    print(f"{'':12} {'n':>4} {'vqa_orig':>9} {'vqa_sent':>9} {'vqa_brand':>9} {'clip_orig':>9}")
    for key in ("by_condition", "by_tier"):
        for k, v in summ[key].items():
            print(f"{k:12} {v['n']:4d} " + " ".join(f"{v[m][0]:9.4f}" for m in
                                                   ("vqa_orig", "vqa_sent", "vqa_brand", "clip_orig")))
    print("\npaired by prompt (injected - control):")
    for m, v in summ["paired_by_prompt"].items():
        print(f"  {m:10} n={v['n_prompts']:2d} mean {v['mean_diff']:+.4f} median {v['median_diff']:+.4f} "
              f"injected higher in {v['injected_higher']}/{v['n_prompts']}  wilcoxon p={v['wilcoxon_p']}")
